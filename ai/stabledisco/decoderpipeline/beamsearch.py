import re

import ai.torchmodules.utils as torchutils
import torch
from clip.clip import _tokenizer as clip_tokenizer

_sot_token = clip_tokenizer.encoder["<|startoftext|>"]
_eot_token = clip_tokenizer.encoder["<|endoftext|>"]

class BeamSearchConfig:
    # Default values are based on a bayesian optimization parameter search
    def __init__(self, model_beams=40, clip_beams=40, num_inter_beams=3000,
                 rating_weight=1.5, clip_weight=8, 
                 ascii_only=True, improve_rating=True, add_evolution_beams=False):
        self.model_beams = model_beams
        self.clip_beams = clip_beams
        self.num_inter_beams = num_inter_beams

        self.rating_weight = rating_weight
        self.clip_weight = clip_weight

        self.ascii_only = ascii_only
        self.improve_rating = improve_rating
        self.add_evolution_beams = add_evolution_beams

class BeamSearcher:
    def __init__(self, tokens_model, ratings_model, clip_model):
        self._tokens_model = tokens_model
        self._ratings_model = ratings_model
        self._clip_model = clip_model

        self._config = BeamSearchConfig()
        
    def set_default_config(self, config):
        self._config = config

    def _get_ascii_mask(self):
        if self._ascii_mask is None:
            self._ascii_mask = torch.ones(len(clip_tokenizer.encoder), device="cuda")
            norm_char_regex = re.compile(
                r"^[a-zA-Z0-9 !\"#$%&'()*+,\-\./:;<=>?@[\]^_`{|}~\\]*$"
            )
            num_ascii = 0
            for token in clip_tokenizer.decoder.keys():
                text = clip_tokenizer.decode([token])
                if norm_char_regex.match(text):
                    num_ascii += 1
                else:
                    self._ascii_mask[token] = 0

        return self._ascii_mask

    def beam_search(
        self,
        features,
        topk=1,
        feature_weights=None,
        config=None,
        verbose=True,
    ):
        if config is None:
            config = self._config  
        epsilon = 1e-8
        self._tokens_model.train(False)
        self._ratings_model.train(False)
        with torch.no_grad():
            if feature_weights is None:
                feature_weights = torch.ones((features.shape[0], 1), device=features.device)

            features = torch.sum(features * feature_weights.view(-1, 1), dim=0)
            features = (features/features.size(0)).unsqueeze(0)
            features_norm = features.norm(dim=-1, keepdim=True)
            features = features / features_norm

            if config.improve_rating:
                features = self._ratings_model.improve_rating(features, verbose=verbose).unsqueeze(0)

            memory = self._tokens_model.features_to_memory(features).squeeze(0)

            torch_zero = torch.tensor([0.0], device=self._tokens_model._device)
            torch_start = torch.tensor([_sot_token], device=self._tokens_model._device, dtype=torch.long)
            curr_beams = [(torch_zero, torch_start)]
            final_beam_tokens = []

            total_beams = config.model_beams + config.clip_beams
            best_match_cosine = torch.tensor(0, device=self._tokens_model._device)

            tokens_to_find = self._tokens_model._seq_len - 2
            for iter_num in range(tokens_to_find):
                curr_tokens = torch.stack(tuple((x[1] for x in curr_beams)))
                token_probs = self._tokens_model.get_next_probs(memory, curr_tokens, ascii_only=config.ascii_only)
                token_probs = torch.log(token_probs + epsilon)

                token_probs = token_probs.size(-1) * token_probs / token_probs.norm(dim=-1, keepdim=True)
                
                new_probs = tuple(
                    curr_beams[idx][0] + token_probs[idx]
                    for idx in range(len(curr_beams))
                )
                next_probs = torch.cat(new_probs).view(-1)

                new_beam_probs, args = next_probs.topk(config.num_inter_beams, dim=-1)
                args = torchutils.unravel_torch_idx(args, next_probs.shape[0] // len(curr_beams))
                prob_beam_token = [(prob, curr_beams[arg[0]][1], arg[1]) for prob, arg in zip(new_beam_probs, args)]

                next_beam_probs = torch.cat(tuple((x[0].unsqueeze(0) for x in prob_beam_token)))
                
                model_arg_sort = torch.argsort(next_beam_probs, dim=-1).cpu().numpy().tolist()
                
                prob_beam_token = [prob_beam_token[idx] for idx in model_arg_sort]

                model_args = model_arg_sort[-config.model_beams:]

                next_beam_tokens = []
                next_beam_tokens_full = torch.zeros(
                    len(prob_beam_token),
                    77,
                    device=self._tokens_model._device,
                    dtype=torch.long,
                )
                idx = 0
                for prob, last_tokens, new_token in prob_beam_token:
                    if len(last_tokens) == self._tokens_model._seq_len - 1:
                        new_token = _eot_token

                    new_token_tensor = torch.tensor([new_token], device=self._tokens_model._device)
                    new_beam = torch.cat((last_tokens,new_token_tensor))
                    next_beam_tokens.append(new_beam)
                    next_beam_tokens_full[idx] = add_token_and_end(last_tokens, new_token)
                    idx += 1

                # The first iteration has the start token place the first selection
                eot_idx = iter_num + 2
                next_beam_cosine_sim_aug = self._clip_model.cosine_similarity(
                    features, next_beam_tokens_full, end_idx=eot_idx, verbosity=2
                )
                cosine_sim, clip_args = next_beam_cosine_sim_aug.topk(total_beams, dim=-1)
                if cosine_sim[0] > best_match_cosine:
                    # TODO: Optimize. Assigning here takes crazy time
                    best_match_cosine = cosine_sim[0]
                    top_tokens = next_beam_tokens_full[clip_args[0]]
                    final_beam_tokens.append(top_tokens)

                if config.clip_weight != 0:
                    clip_bonus = torch.softmax(next_beam_cosine_sim_aug, dim=-1)
                    clip_bonus = torch.log(clip_bonus + epsilon) * config.clip_weight
                    new_beam_probs += clip_bonus

                if config.rating_weight != 0:
                    beam_features = self._clip_model.features_from_tokens(next_beam_tokens_full)
                    aug_ratings = self._ratings_model(beam_features).view(-1)
                    aug_rating_probs = torch.softmax(aug_ratings, dim=-1)
                    aug_ratings_bonus = (torch.log(aug_rating_probs + epsilon) * config.rating_weight)
                    new_beam_probs += aug_ratings_bonus

                _, model_args = (new_beam_probs).topk(config.model_beams, dim=-1)

                clip_args = clip_args.cpu().numpy()
                clip_args = [arg for arg in clip_args if arg not in model_args and prob_beam_token[arg][0] != _eot_token][:config.clip_beams]
                top_args = model_args.cpu().numpy().tolist() + clip_args
                curr_beams = []

                for top_idx in top_args:
                    prob = prob_beam_token[top_idx][0]
                    new_beam = next_beam_tokens[top_idx]
                    curr_beams.append((prob, new_beam))
                    if config.add_evolution_beams:
                        # Used for experimental evolutionary algorithms
                        final_beam_tokens.append(next_beam_tokens_full[top_idx])

                if verbose and (iter_num % 10 == 0 or iter_num == tokens_to_find - 1):
                    print(f"{iter_num} of {self._tokens_model._seq_len} tokens searched")
                    best_features = self._clip_model.features_from_tokens(final_beam_tokens[-1].view(1, -1))
                    rating = self._ratings_model(best_features)[0].item()
                    print(f"curr top {best_match_cosine: 0.4f} with estimated quality {rating: 0.2f}: {self._tokens_model.decode(final_beam_tokens[-1])}"
)

            top_k = min(topk, len(final_beam_tokens))
            final_beam_tokens = torch.stack(tuple(final_beam_tokens)).long()
            best_tokens, final_cosine_sim = self._clip_model.rank_similarity(features, final_beam_tokens, top_k)

            return self._tokens_model.decode(best_tokens), final_cosine_sim


def add_token_and_end(curr_tokens, new_token, max_size=77):
    if curr_tokens.size(0) >= max_size:
        return curr_tokens

    device = curr_tokens.get_device()

    possible_token = torch.tensor([new_token], device=device, dtype=torch.long)
    if new_token == _eot_token:
        ret = torch.cat((curr_tokens, possible_token))
    else:
        ret = torch.cat(
            (
                curr_tokens,
                possible_token,
                torch.tensor([_eot_token], device=device, dtype=torch.long),
            )
        )

    rem_tokens = max(max_size - ret.size(0), 0)
    if rem_tokens > 0:
        ret = torch.cat((ret, torch.zeros(rem_tokens, device=device, dtype=torch.long)))

    return ret
