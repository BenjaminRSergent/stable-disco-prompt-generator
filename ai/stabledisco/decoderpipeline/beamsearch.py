import re

import ai.stabledisco.constants as sdconsts
import ai.stabledisco.utils as sdutils
import ai.torchmodules.utils as torchutils
import clip
import torch
from ai.stabledisco import decoderpipeline
from ai.stabledisco.clipmodel import ClipModel
from ai.stabledisco.decoderpipeline.featurestorating import \
    FeaturesToRatingModel
from ai.stabledisco.decoderpipeline.featurestotokensaes import \
    FeaturesToTokensAesModel
from clip.clip import _tokenizer as clip_tokenizer


class UpgradeConfig:
    def __init__(self,
                num_passes = 6,
                num_large_passes = 8,
                baseline_cands = 768*2,
                min_cands = 256,
                large_pass_mul = 1,
                token_step_size = 20,
                tokens_per_pass = 10,
                upgrade_iter_start = 27,
                large_pass_freq = 20):
        self.num_passes = num_passes
        self.num_large_passes = num_large_passes
        self.baseline_cands = baseline_cands
        self.min_cands = min_cands
        self.large_pass_mul = large_pass_mul
        self.token_step_size = token_step_size 
        self.tokens_per_pass = tokens_per_pass 
        self.upgrade_iter_start = upgrade_iter_start 
        self.large_pass_freq = large_pass_freq

class BeamSearchConfig:
    # Default values are based on a bayesian optimization parameter search
    def __init__(self, model_beams=40, clip_beams=40, num_inter_beams=3500,
                 rating_weight=1.0, clip_weight=2, 
                 enable_upgrades=True,
                 ascii_only=True, add_evolution_beams=False,
                 device=None, verbose=True):
        self.model_beams = model_beams
        self.clip_beams = clip_beams
        self.num_inter_beams = num_inter_beams

        self.enable_upgrades = enable_upgrades
        self.rating_weight = rating_weight
        self.clip_weight = clip_weight
        self.ascii_only = ascii_only
        self.add_evolution_beams = add_evolution_beams
        self.verbose = verbose

        
        
        if device is None:
            device = torchutils.get_default_device()
        self.device = device

class BeamSearcher:
    _epsilon = 1e-6
    class BeamSearchState:
        # Variables related to a given beam search to reduce excessive method parameters
        def __init__(self, features: torch.tensor, memory: torch.tensor, config: BeamSearchConfig, upgrade_config: UpgradeConfig):
            self.features = features
            self.memory = memory
            torch_zero = torch.tensor([0.0], device=config.device)
            torch_start = torch.tensor([sdconsts.sot_token], device=config.device, dtype=torch.long)
            self.curr_beams = [(torch_zero, torch_start)]
            self.final_beam_tokens = []

            self.total_beams = config.model_beams + config.clip_beams
            self.best_match_cosine = torch.tensor(0, device=config.device)
            self.iter_without_improvement = 0
            self.config = config
            self.upgrade_config = upgrade_config
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
            if self.next_token == sdconsts.eot_token:
                # The next token is already EOT, only add it
                ret = torch.cat((self.prev_tokens, possible_token))
            else:
                # Add the new token and an EOT token
                ret = torch.cat(
                    (
                        self.prev_tokens,
                        possible_token,
                        torch.tensor([sdconsts.eot_token], device=device, dtype=torch.long),
                    )
                )

            rem_tokens = max(max_size - ret.size(0), 0)
            if rem_tokens > 0:
                # Add padding for the remaining size
                ret = torch.cat((ret, torch.zeros(rem_tokens, device=device, dtype=torch.long)))

            return ret
            
    def __init__(self, tokens_model: FeaturesToTokensAesModel, ratings_model: FeaturesToRatingModel, clip_model: ClipModel, search_config=None, upgrade_config=None):
        self._tokens_model = tokens_model
        self._ratings_model = ratings_model
        self._clip_model = clip_model
        self._upgrader = decoderpipeline.PromptUpgrader(self._tokens_model, clip_model, self._ratings_model, verbose=False)
        if search_config is None:
            search_config = BeamSearchConfig()
        self._default_search_config = search_config

        if upgrade_config is None:
            upgrade_config = UpgradeConfig()
        self._default_upgrade_config = upgrade_config
        
    def set_default_search_config(self, config):
        self._default_search_config = config

    def set_default_upgrade_config(self, config):
        self._default_upgrade_config = config

    def beam_search(self, features, max_len=75, topk=1, search_config=None, upgrade_config=None, verbose=True):
        if search_config is None:
            search_config = self._default_search_config  
        if upgrade_config is None:
            upgrade_config = self._default_upgrade_config  

        self._tokens_model.train(False)
        self._ratings_model.train(False)

        if verbose:
            print(f"Starting beam search to find the top {topk} prompts of length {max_len}.")

        with torch.no_grad():
            # Setup the Inital search state
            features = self._process_orig_features(features)
            memory = self._tokens_model.features_to_memory(features).squeeze(0)
            search_state = BeamSearcher.BeamSearchState(features, memory, search_config, upgrade_config)

            tokens_to_find = min(max_len, self._tokens_model._seq_len - 2)
            for _ in range(tokens_to_find):
                    
                self._do_beam_search_step(search_state)

                if verbose and search_state.iter_num % 1 == 0:
                    self._print_search_state(search_state)
                    
                search_state.iter_num += 1
            
            if verbose:
                self._print_search_state(search_state)
            
            # Choose k tokens with the highest estimated score
            best_tokens, final_cosine_sim = self._get_final_tokens(search_state, topk)

            return self._tokens_model.decode(best_tokens), final_cosine_sim, search_state.features
    
    def _process_orig_features(self, features):
        # Take a weighted sum of the features and normalize
        features = features.view(sdconsts.feature_width)
        features = (features/features.size(0)).unsqueeze(0)
        features_norm = features.norm(dim=-1, keepdim=True)
        features = features / features_norm

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

        
        
        search_state.curr_beams = []
        self._upgrade_top(search_state, candidates, top_clip_idxs, top_beam, next_beam_tokens_full)
        # TODO: profile unquie versus doing duplicates
        top_idxs = torch.unique(torch.cat((top_clip,top_beam))).cpu().tolist()
        # Append the choosen tokens to their parent beam for the next iteration
        for top_idx in top_idxs:
            prob = candidates[top_idx].prob
            new_beam = next_beam_tokens[top_idx]
            
            
            search_state.curr_beams.append((prob, new_beam))
            add_evolution(next_beam_tokens_full[top_idx])

    def _upgrade_top(self, search_state, candidates, top_clip_idxs, top_beam_idxs, next_beam_tokens_full):
        upgrade_config = search_state.upgrade_config

        tokens_added = candidates[0].prev_tokens.size(0)+1
        tokens_since_start = tokens_added - upgrade_config.upgrade_iter_start
        num_passes = upgrade_config.num_passes
        #upgrade_top = tokens_since_start > 0 and (tokens_since_start % upgrade_config.tokens_per_pass) == 0
        upgrade_top = search_state.iter_without_improvement > 0
        
        
        start_idx = (upgrade_config.token_step_size * search_state.times_upgraded) % (tokens_added-1)
        start_idx = max(1, start_idx)

        if start_idx < upgrade_config.token_step_size:
            # Avoid skipping beginning tokens after wrapping around
            start_idx = 1

        end_idx = start_idx + upgrade_config.token_step_size

        start_to_end = (73 - upgrade_config.upgrade_iter_start)

        # Linearly decrease candidates to 1024
        num_candidates = max(int(upgrade_config.baseline_cands * (1 - tokens_since_start/start_to_end)), upgrade_config.min_cands)

        # Extra passes the first time it goes through each range
        if tokens_since_start >= 0 and tokens_since_start % upgrade_config.large_pass_freq == 0:
            start_idx = 1
            end_idx = -1
            num_candidates = int(upgrade_config.baseline_cands*upgrade_config.large_pass_mul)
            num_passes = upgrade_config.num_large_passes
            upgrade_top = True
            
        top_clip_beams = next_beam_tokens_full[top_clip_idxs]
            
        top_beam, cosine_sim = self._clip_model.rank_similarity(search_state.features, top_clip_beams, top_count=1)
        top_beam = top_beam[0]
        cosine_sim = torch.tensor(cosine_sim[0], device= top_beam.device)

        
        upgrade_top = cosine_sim - search_state.best_match_cosine <= self._epsilon
        cands_mul = 4
        min_cands = num_candidates
        max_cands = min_cands * cands_mul ** 2
        first_pass = max_cands * 1.5
        num_candidates = first_pass 
        con_improve = 0
        con_improve_to_div = 2
        
        times_wrapped = 0
        max_wraps = 4
        if upgrade_top and search_state.config.enable_upgrades:
            to_upgrade = [top_clip_idxs[0]]

            if top_clip_idxs[0] != top_beam_idxs[0]:
                to_upgrade.append(top_beam_idxs[0])

            for idx in to_upgrade:#, top_beam_idxs[0]]:
            
                orig_beam = top_beam
                beam_eot_idx = sdutils.find_end_idx(orig_beam)
                new_beam = orig_beam.clone()

                last_score = self._upgrader._calculator.score_tokens(search_state.features, new_beam)[0]
                for pass_num in range(num_passes):
                    print(f"Running up to {num_passes} upgrade passes on {tokens_added-2} tokens at {num_candidates} from {start_idx} to {end_idx} on beam with the highest similarity. div {con_improve}")
                    new_beam, new_score = self._upgrader.single_upgrade_pass(search_state.features, new_beam, start_idx=start_idx, end_idx=end_idx,
                                                                            num_candidates=int(num_candidates), ascii_only=search_state.config.ascii_only,
                                                                            decay_factor=0.85)
                    if (new_score - last_score) < self._epsilon:
                        con_improve = 0
                        if num_candidates < max_cands:
                            num_candidates *= cands_mul
                        else:
                            times_wrapped += 1
                            if times_wrapped >= max_wraps:
                                break
                            else:
                                num_candidates = first_pass 
                                start_idx += upgrade_config.token_step_size
                                end_idx += upgrade_config.token_step_size
                                if start_idx >= tokens_added:
                                    start_idx -= tokens_added
                                    end_idx -= tokens_added
                                     
                    elif num_candidates > max_cands:
                        num_candidates = min_cands
                    elif num_candidates > min_cands:
                        con_improve += 1
                        if con_improve >= con_improve_to_div:
                            con_improve = 0
                            num_candidates = max(min_cands, num_candidates//cands_mul)
                        
                    if search_state.config.verbose:
                        print(f"Pass {pass_num} increased score from {last_score:0.3f} to {new_score:0.3f}")
                        last_score = new_score

                upgraded_features = self._clip_model.features_from_tokens(new_beam.view(1, -1))
                cosine_sim = self._clip_model.cosine_similarity(search_state.features, upgraded_features)[0]
                if cosine_sim > search_state.best_match_cosine:
                    search_state.best_match_cosine = cosine_sim
                    search_state.final_beam_tokens.append(new_beam)

                
                if search_state.config.verbose:
                    orig_features = self._clip_model.features_from_tokens(orig_beam.view(1, -1))
                    orig_sim = self._clip_model.cosine_similarity(search_state.features, orig_features)[0]
                    orig_rating = self._ratings_model(orig_features)[0].item()

                    upgraded_rating = self._ratings_model(upgraded_features)[0].item()
                    print(f"Upgraded changed cosine sim from {orig_sim} to {cosine_sim: 0.3f} and estimated quality from {orig_rating: 0.3f} to {upgraded_rating: 0.3f}:\n {self._tokens_model.decode(new_beam)[0]}")

                top_beam = new_beam
                new_beam = new_beam[:beam_eot_idx]
                
                search_state.curr_beams.append((candidates[idx].prob, new_beam.clone()))

            search_state.times_upgraded += 1
            
            search_state.iter_without_improvement = -1
            
        if cosine_sim > search_state.best_match_cosine:
            search_state.best_match_cosine = cosine_sim
            search_state.final_beam_tokens.append(top_beam)

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
                candidate.next_token = sdconsts.eot_token

            next_beam_tokens.append(candidate.get_beam_tokens())
            next_beam_tokens_full[idx] = candidate.get_beam_tokens_with_end()
            idx += 1

        return next_beam_tokens, next_beam_tokens_full

    def _get_top_cosine_sim(self, search_state, next_beam_tokens_full):
        # The first iteration has two tokens before the end: the start token place the first selection
        eot_idx = search_state.iter_num + 2
        # TODO:Rank
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
        
        unique_cnt = torch.unique(torch.cat((top_beam_idxs, top_clip_idxs[:search_state.config.clip_beams]))).size(0)
        
        overlap = search_state.config.clip_beams+search_state.config.model_beams-unique_cnt

        return top_clip_idxs[:search_state.config.clip_beams+overlap], top_beam_idxs

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
        
        
        #Check for retokenization inconsistencies. This can occur when multiple token sequences can represent the same prompt
        orig = self._clip_model.cosine_similarity(search_state.features, [search_state.final_beam_tokens[-1]])[0]

        telephone_tokens = clip.tokenize(self._tokens_model.decode(search_state.final_beam_tokens[-1]))[0].cuda()
        telephone = self._clip_model.cosine_similarity(search_state.features, [telephone_tokens])[0]
        """
        print(f"Cosine telephone {search_state.best_match_cosine: 0.4f} recalc {orig:0.4f} to {telephone: 0.4f}")
        print("Tokens pre", search_state.final_beam_tokens[-1])
        print("Tokens post", telephone_tokens)
        """
        print(f"Top beam has cosine sim {search_state.best_match_cosine: 0.3f} with estimated quality of {rating: 0.3f}:\n {self._tokens_model.decode(search_state.final_beam_tokens[-1])[0]}")


    def _safe_log(self, tensor):
        return torch.log(tensor + self._epsilon)


