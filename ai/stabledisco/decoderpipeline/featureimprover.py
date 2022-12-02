from functools import lru_cache
import ai.stabledisco.utils as sdutils
import torch


class FeatureImprover:
    def __init__(self, is_text_model, to_rating_model):
        self._is_text_model = is_text_model
        self._to_rating_model = to_rating_model

    @lru_cache(maxsize=32)
    def improve_features_v2(
        self,
        orig_features,
        alpha=0.75,
        target_rating=9.0,
        target_prob=1.0,
        prob_scale=1.25,
        rating_scale=1,
        sim_scale=1.5,
        max_diff=0.1,
        step_size=1e-1,
        num_steps=200,
        verbose=True,
    ):

        if verbose:
            print("Starting upgrade")

        with torch.no_grad():
            orig_features = orig_features / orig_features.norm(dim=-1, keepdim=False)

        self._to_rating_model.freeze()
        self._is_text_model.freeze()

        ret_features = torch.nn.Parameter(orig_features.clone(), requires_grad=True)
        ret_features.requires_grad_(True)
        orig_features.requires_grad_(True)
        optim = torch.optim.NAdam([ret_features], lr=step_size)

        # Place the rating in [0, 1] to match the other features
        def normalize_rating(rating):
            return (rating - 7) / 3

        rating_target = torch.tensor([normalize_rating(target_rating)], dtype=torch.float, device=orig_features.device)
        text_target = torch.tensor([target_prob], dtype=torch.float, device=orig_features.device)
        sim_target = torch.tensor([sim_scale], dtype=torch.float, device=orig_features.device)

        mse = torch.nn.MSELoss()

        def get_score(other):
            rating_norm = normalize_rating(self._to_rating_model.get_rating(other)).view(1)
            rating_score = rating_scale * mse(rating_norm, rating_target)

            text_score = torch.minimum(self._is_text_model.get_text_prob(other), text_target).view(1)
            text_score = prob_scale * mse(text_score, text_target)

            sim_score = (sdutils.cosine_sim(orig_features, other) * sim_scale).view(1)
            sim_score = sim_scale * mse(sim_score, sim_target)

            return (rating_score + text_score + sim_score).view((1, 1))

        ret_features = torch.nn.Parameter(orig_features.clone(), requires_grad=True)
        ret_features.requires_grad_(True)
        optim = torch.optim.SGD([ret_features], lr=step_size)

        best_loss = get_score(ret_features)
        best_features = ret_features.clone()

        for step_num in range(num_steps):
            if verbose and step_num % 250 == 0:
                print(f"Upgrade step {step_num}")
                after_prob = self._is_text_model.get_text_prob(best_features)
                print(f"Text probability: {after_prob.item()}")
                after_rating = self._to_rating_model.get_rating(best_features)
                print(f"Feature rating: {after_rating.item()}")
            pre_step = ret_features.clone()
            loss = get_score(ret_features)
            loss.backward()
            optim.step()
            optim.zero_grad()
            with torch.no_grad():
                if loss < best_loss:
                    norm_features = ret_features / ret_features.norm(keepdim=True)
                    cos_diff = 1 - sdutils.cosine_sim(norm_features, orig_features)
                    if cos_diff > max_diff:
                        ret_features = pre_step
                        step_size *= alpha
                    else:
                        best_loss = loss
                        best_features = norm_features.clone()
                else:
                    step_size *= alpha

        with torch.no_grad():
            best_features /= best_features.norm(keepdim=True)

        if verbose:
            with torch.no_grad():
                before_prob = self._is_text_model.get_text_prob(orig_features)
                after_prob = self._is_text_model.get_text_prob(best_features)
                print(f"Text probability change: {before_prob.item()} -> {after_prob.item()}")

                before_rating = self._to_rating_model.get_rating(orig_features)
                after_rating = self._to_rating_model.get_rating(best_features)
                print(f"Feature rating change: {before_rating.item()} -> {after_rating.item()}")
                print(f"Cosine Diff: {1 - sdutils.cosine_sim(orig_features, best_features )}")
                print()

        return best_features

    def improve_features(
        self, orig_features, target_rating=9.0, target_prob=1.0, max_rating_diff=0.1, max_prob_diff=0.15, verbose=True
    ):

        with torch.no_grad():
            # TODO: Combined improvements
            if len(orig_features.shape) == 1:
                orig_features = orig_features.unsqueeze(0)
            orig_features = sdutils.norm_t(orig_features)
            target_features = orig_features
            target_features = self._is_text_model.improve_text_prob(
                target_features, target_prob=target_prob, max_diff=max_prob_diff, verbose=verbose
            ).view(1, -1)
            target_features = self._to_rating_model.improve_rating_v2(
                target_features, max_diff=max_rating_diff, target_rating=target_rating, verbose=verbose
            ).view(1, -1)
            target_features = self._is_text_model.improve_text_prob(
                target_features, target_prob=target_prob, max_diff=max_prob_diff / 2, verbose=verbose
            ).view(1, -1)

            if verbose:
                before_prob = self._is_text_model.get_text_prob(orig_features)
                after_prob = self._is_text_model.get_text_prob(target_features)
                print(f"Text probability change: {before_prob} -> {after_prob}")

                before_rating = self._to_rating_model.get_rating(orig_features)
                after_rating = self._to_rating_model.get_rating(target_features)
                print(f"Feature rating change: {before_rating} -> {after_rating}")
                print(f"Cosine diff: {1 - sdutils.cosine_sim(orig_features,target_features )}")
                print()

        return target_features
