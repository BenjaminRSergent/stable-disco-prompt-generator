import re

import ai.torchmodules.utils as torchutils
import torch
from ai.stabledisco import decoderpipeline
from ai.stabledisco.clipmodel import ClipModel
from ai.stabledisco.decoderpipeline.featurestorating import \
    FeaturesToRatingModel
from ai.stabledisco.decoderpipeline.featurestotokensaes import \
    FeaturesToTokensAesModel
from clip.clip import _tokenizer as clip_tokenizer

_sot_token = clip_tokenizer.encoder["<|startoftext|>"]
_eot_token = clip_tokenizer.encoder["<|endoftext|>"]


class BeamSearchConfig:
    # Default values are based on a bayesian optimization parameter search
    def __init__(self, model_beams=40, clip_beams=40, num_inter_beams=3500,
                 rating_weight=1.0, clip_weight=2, 
                 ascii_only=True, improve_rating=True, rating_max_diff=0.025, add_evolution_beams=False,
                 device=None):
        self.model_beams = model_beams
        self.clip_beams = clip_beams
        self.num_inter_beams = num_inter_beams

        self.rating_weight = rating_weight
        self.clip_weight = clip_weight
        self.ascii_only = ascii_only
        self.improve_rating = improve_rating
        self.rating_max_diff = rating_max_diff
        self.add_evolution_beams = add_evolution_beams
        
        if device is None:
            device = torchutils.get_default_device()
        self.device = device

class BeamSearcher:
    _epsilon = 1e-6
    class BeamSearchState:
        # Variables related to a given beam search to reduce excessive method parameters
        def __init__(self, features: torch.tensor, memory: torch.tensor, config: BeamSearchConfig):
            self.features = features
            self.memory = memory
            torch_zero = torch.tensor([0.0], device=config.device)
            torch_start = torch.tensor([_sot_token], device=config.device, dtype=torch.long)
            self.curr_beams = [(torch_zero, torch_start)]
            self.final_beam_tokens = []

            self.total_beams = config.model_beams + config.clip_beams
            self.best_match_cosine = torch.tensor(0, device=config.device)
            self.config = config
            self.iter_num = 0
            self.times_upgraded = 0

            self.next_beam_scores = None

    class CandidateBeam:
        # A possible next token for a beam with the estimated probability that it is the next token
        def __init__(self, prob, prev_tokens, next_token):
            self.prob = prob
            self.prev_tokens = prev_tokens
            self.next_token = next_token

        def get_beam_tokens(self):
            new_token_tensor = torch.tensor([self.next_token], device=self.prev_tokens.device)
            return torch.cat((self.prev_tokens, new_token_tensor))

        def get_beam_tokens_with_end(self, max_size=77):
            # Add an EOT token and padding
            if self.prev_tokens.size(0) >= max_size:
                return self.prev_tokens

            device = self.prev_tokens.get_device()
            possible_token = torch.tensor([self.next_token], device=device, dtype=torch.long)
            if self.next_token == _eot_token:
                # The next token is already EOT, only add it
                ret = torch.cat((self.prev_tokens, possible_token))
            else:
                # Add the new token and an EOT token
                ret = torch.cat(
                    (
                        self.prev_tokens,
                        possible_token,
                        torch.tensor([_eot_token], device=device, dtype=torch.long),
                    )
                )

            rem_tokens = max(max_size - ret.size(0), 0)
            if rem_tokens > 0:
                # Add padding for the remaining size
                ret = torch.cat((ret, torch.zeros(rem_tokens, device=device, dtype=torch.long)))

            return ret
            
    def __init__(self, tokens_model: FeaturesToTokensAesModel, ratings_model: FeaturesToRatingModel, clip_model: ClipModel, config=None):
        self._tokens_model = tokens_model
        self._ratings_model = ratings_model
        self._clip_model = clip_model
        self._upgrader = decoderpipeline.PromptUpgrader(self._tokens_model, clip_model, self._ratings_model, verbose=False)
        if config is None:
            config = BeamSearchConfig()
        self._default_config = config
        
    def set_default_config(self, config):
        self._default_config = config

    def beam_search(self, features, max_len=77, topk=1, feature_weights=None, config=None, verbose=True):
        if config is None:
            config = self._default_config  

        self._tokens_model.train(False)
        self._ratings_model.train(False)

        with torch.no_grad():
            # Setup the Inital search state
            features = self._process_orig_features(features, feature_weights, config, verbose)
            memory = self._tokens_model.features_to_memory(features).squeeze(0)
            search_state = BeamSearcher.BeamSearchState(features, memory, config)

            tokens_to_find = min(max_len, self._tokens_model._seq_len - 2)
            for _ in range(tokens_to_find):
                self._do_beam_search_step(search_state)

                if verbose and search_state.iter_num % 5 == 0:
                    self._print_search_state(search_state)
                    
                search_state.iter_num += 1
            
            if verbose:
                self._print_search_state(search_state)
            
            # Choose k tokens with the highest estimated score
            best_tokens, final_cosine_sim = self._get_final_tokens(search_state, topk)

            return self._tokens_model.decode(best_tokens), final_cosine_sim, search_state.features
    
    def _process_orig_features(self, features, feature_weights, config, verbose):
        if feature_weights is None:
            # Default to equally weighting each feature tensor
            feature_weights = torch.ones((features.shape[0], 1), device=features.device)

        # Take a weighted sum of the features and normalize
        features = torch.sum(features * feature_weights.view(-1, 1), dim=0)
        features = (features/features.size(0)).unsqueeze(0)
        features_norm = features.norm(dim=-1, keepdim=True)
        features = features / features_norm

        if config.improve_rating:
            # Move the features slightly toward parts of the latent space estimated to be higher rated.
            # Small nudges result in higher accuracy since the error tends to be in the direction of lower rated
            # parts of the latent space.
            features = self._ratings_model.improve_rating(features, max_diff=config.rating_max_diff, verbose=verbose).unsqueeze(0)

        return features

    def _do_beam_search_step(self, search_state: BeamSearchState):
        # Narrow the search space to tokens that the model estimates are likely good matches
        candidates = self._get_candidate_beams(search_state)

        # Create new token tensors based on the candidates
        next_beam_tokens, next_beam_tokens_full = self._get_next_tokens(search_state, candidates)

        # Choose tokens based on their cosine similarity with the features
        top_clip_idxs = self._get_top_cosine_sim(search_state, next_beam_tokens_full)

        if search_state.config.rating_weight != 0:
            # Nudge the selection process toward tokens estimated to increase the quality of the result
            self._apply_rating_bonus(search_state, next_beam_tokens_full)
        
        # Choose tokens with high estimated scores and high cosine similarity with the featurs
        top_clip, top_beam = self._get_top_idxs(search_state, top_clip_idxs)

        added = 0
        candidates_per_iter = 10
        def add_evolution(full_beam):
            nonlocal added
            if search_state.config.add_evolution_beams and added < candidates_per_iter:
                # Used for experimental evolutionary algorithms. Intermediate beams during the search
                # are useful for the initial population
                search_state.final_beam_tokens.insert(0, full_beam)
                added += 1

        
        tokens_added = candidates[0].prev_tokens.size(0)+1
        search_state.curr_beams = []

        num_passes = 5
        token_step_size = 10

        tokens_per_pass = 4
        upgrade_iter_start = 16
        upgrade_top = tokens_added >= upgrade_iter_start and (tokens_added % tokens_per_pass) == 0

        start_idx = (token_step_size * (tokens_added - upgrade_iter_start)//tokens_per_pass) % (tokens_added-1)
        start_idx = max(1, start_idx)

        if start_idx < token_step_size:
            # Avoid skipping beginning tokens after wrapping around
            start_idx = 1

        end_idx = start_idx + token_step_size

        start_to_end = (73 - upgrade_iter_start)
        since_start = (tokens_added - upgrade_iter_start)

        # Linearly decrease candidates to 1024
        num_candidates = max(int(768 * (1 - since_start/start_to_end)), 128)

        
        # Extra passes the first time it goes through each range
        if tokens_added % 16 == 0:
            start_idx = 1
            end_idx = -1
            num_candidates = int(num_candidates*3)
            num_passes = 10

        if tokens_added == 73:
            start_idx = 50
            end_idx = -1
            num_candidates = 512*3
            upgrade_top = True
        elif tokens_added == 74:
            start_idx = 25
            end_idx = 50
            num_candidates = 512*3
            upgrade_top = True
        elif tokens_added == 75:
            start_idx = 1
            end_idx = -1
            num_candidates = 1024
            num_passes = 10
            upgrade_top = True


        if upgrade_top:
            # TODO: Replace bottom beam with an upgraded version of the top beam

            print(f"Running {num_passes} upgrade passes on {tokens_added-2} tokens at {num_candidates} from {start_idx} to {end_idx}")
            full_beam = next_beam_tokens_full[top_clip[0]]
            beam_eot_idx = torch.argwhere(full_beam == _eot_token).view(-1)[0]
            new_beam = full_beam
            
            last_score = self._upgrader._calculator.score_tokens(search_state.features, new_beam)[0]
            #print(f"Start {start_idx}, end {end_idx}, cand {num_candidates}\n {new_beam}")
            for _ in range(num_passes):
                new_beam, new_score = self._upgrader.single_upgrade_pass(search_state.features, new_beam, start_idx=start_idx, end_idx=end_idx,
                                                                         num_candidates=num_candidates, ascii_only=search_state.config.ascii_only,
                                                                         decay_factor=0.85)
                if (new_score - last_score) < 1e-2:
                    break
                print(f"{last_score} vs {new_score}")
                last_score = new_score

            cosine_sim = self._clip_model.cosine_similarity(search_state.features, [new_beam])
            if cosine_sim[0] > search_state.best_match_cosine:
                search_state.best_match_cosine = cosine_sim[0]
                search_state.final_beam_tokens.append(new_beam)

            add_evolution(new_beam)
            print(self._tokens_model.decode(new_beam))
            new_beam = new_beam[:beam_eot_idx]
            # Make the upgraded beam tie with the top probability to focus on it more
            search_state.curr_beams.append((candidates[top_beam[0]].prob, new_beam))

            search_state.times_upgraded += 1
        else:
            cosine_sim = self._clip_model.cosine_similarity(search_state.features, next_beam_tokens_full[top_clip[0]])
            if cosine_sim[0] > search_state.best_match_cosine:
                search_state.best_match_cosine = cosine_sim[0]
                search_state.final_beam_tokens.append(next_beam_tokens_full[top_clip[0]])

        top_idxs = top_clip+top_beam
        # Append the choosen tokens to their parent beam for the next iteration
        for top_idx in top_idxs:
            prob = candidates[top_idx].prob
            new_beam = next_beam_tokens[top_idx]
            
            
            search_state.curr_beams.append((prob, new_beam))
            add_evolution(next_beam_tokens_full[top_idx])

    def _get_candidate_beams(self, search_state):
        # Get the model's estimate for the next token's probability for each beam
        curr_tokens = torch.stack(tuple((x[1] for x in search_state.curr_beams)))
        token_probs = self._tokens_model.get_next_probs(search_state.memory, curr_tokens, ascii_only=search_state.config.ascii_only)
        token_probs = self._safe_log(torch.relu(token_probs))
        token_probs = token_probs / token_probs.norm(dim=-1, keepdim=True)

        # Add the log of the token's probablity to its parent's cumulative probability
        new_probs = tuple(
            search_state.curr_beams[idx][0] + token_probs[idx]
            for idx in range(len(search_state.curr_beams))
        )

        # Find next token choices with the highest cumulative probability as 1D indices
        next_probs = torch.cat(new_probs).view(-1)
        next_beam_scores, candidate_idxs = next_probs.topk(search_state.config.num_inter_beams, dim=-1)
        search_state.next_beam_scores = next_beam_scores

        # Convert the 1D indices into 2D to associate each with their parent to get their previous tokens
        candidate_idxs = torchutils.unravel_torch_idx(candidate_idxs, len(clip_tokenizer.encoder))

        def make_candidate_beam(prob, idx):
            return BeamSearcher.CandidateBeam(prob, search_state.curr_beams[idx[0]][1], idx[1])

        candidates = [make_candidate_beam(prob, idx) for prob, idx in zip(next_beam_scores, candidate_idxs)]

        # Later steps in the iteration require the candidates to be sorted
        next_beam_probs = torch.cat(tuple((candidate.prob.unsqueeze(0) for candidate in candidates)))
        model_arg_sort = torch.argsort(next_beam_probs, dim=-1).cpu().numpy().tolist()
        candidates = [candidates[idx] for idx in model_arg_sort]

        return candidates

    def _get_next_tokens(self, search_state, candidates):
        # Create a list of tokens for each beams and a tensor of those tokens with an EOT token and padding
        next_beam_tokens = []
        next_beam_tokens_full = torch.zeros(
            len(candidates),
            77,
            device=search_state.config.device,
            dtype=torch.long,
        )
        idx = 0
        for candidate in candidates:
            if len(candidate.prev_tokens) == self._tokens_model._seq_len - 1:
                candidate.next_token = _eot_token

            next_beam_tokens.append(candidate.get_beam_tokens())
            next_beam_tokens_full[idx] = candidate.get_beam_tokens_with_end()
            idx += 1

        return next_beam_tokens, next_beam_tokens_full

    def _get_top_cosine_sim(self, search_state, next_beam_tokens_full):
        # The first iteration has two tokens before the end: the start token place the first selection
        eot_idx = search_state.iter_num + 2
        next_beam_cosine_sim_aug = self._clip_model.cosine_similarity(
            search_state.features, next_beam_tokens_full, end_idx=eot_idx, verbosity=2
        )
        _, top_clip_idxs = next_beam_cosine_sim_aug.topk(search_state.total_beams, dim=-1)


        if search_state.config.clip_weight != 0:
            clip_bonus = torch.softmax(next_beam_cosine_sim_aug, dim=-1)
            clip_bonus = self._safe_log(clip_bonus) * search_state.config.clip_weight
            search_state.next_beam_scores += clip_bonus

        return top_clip_idxs

    def _apply_rating_bonus(self, search_state, next_beam_tokens_full):
        beam_features = self._clip_model.features_from_tokens(next_beam_tokens_full)
        ratings = self._ratings_model(beam_features).view(-1)
        ratings_score = torch.softmax(ratings, dim=-1)
        aug_ratings_bonus = self._safe_log(ratings_score) * search_state.config.rating_weight
        search_state.next_beam_scores += aug_ratings_bonus

    def _get_top_idxs(self, search_state, top_clip_idxs):
        _, top_beam_idxs = search_state.next_beam_scores.topk(search_state.config.model_beams, dim=-1)

        top_clip_idxs = top_clip_idxs.cpu().tolist()
        top_clip_idxs = [idx for idx in top_clip_idxs if idx not in top_beam_idxs][:search_state.config.clip_beams]
        return top_clip_idxs, top_beam_idxs.cpu().tolist()

    def _get_final_tokens(self, search_state, topk):
        topk = min(topk, len(search_state.final_beam_tokens))
        search_state.final_beam_tokens = torch.stack(tuple(search_state.final_beam_tokens)).long()
        return self._clip_model.rank_similarity(search_state.features, search_state.final_beam_tokens, topk)

    def _print_search_state(self, search_state):
        if not search_state.final_beam_tokens:
            return
        print(f"{search_state.iter_num} of {self._tokens_model._seq_len-2} tokens searched")
        best_features = self._clip_model.features_from_tokens(search_state.final_beam_tokens[-1].view(1, -1))
        rating = self._ratings_model(best_features)[0].item()
        print(f"curr top {search_state.best_match_cosine: 0.4f} with estimated quality {rating: 0.2f}: {self._tokens_model.decode(search_state.final_beam_tokens[-1])}")

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

    def _safe_log(self, tensor):
        return torch.log(tensor + self._epsilon)


