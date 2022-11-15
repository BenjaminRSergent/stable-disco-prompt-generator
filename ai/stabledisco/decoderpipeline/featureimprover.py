import ai.stabledisco.utils as sdutils
import torch


class FeatureImprover:
    def __init__(self, is_text_model, to_rating_model):
        self._is_text_model = is_text_model
        self._to_rating_model = to_rating_model

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
            target_features = self._to_rating_model.improve_rating(
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
