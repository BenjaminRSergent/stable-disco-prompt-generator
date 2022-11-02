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
import copy

class UpgradeConfig:
    def __init__(self,
                baseline_cands=1024,
                upgrade_iter_start=10,
                upgrade_threshold= 7,
                cap_margin=4,
                verbose=True):
        self.baseline_cands = baseline_cands
        self.upgrade_iter_start = upgrade_iter_start 
        self.upgrade_threshold = upgrade_threshold
        self.cap_margin = cap_margin
        self.verbose = verbose

class BeamSearchConfig:
    # Default values are based on a bayesian optimization parameter search
    def __init__(self, model_beams=60, clip_beams=1000,
                 num_inter_beams=1024, max_inter_beams=1024*3,
                 rating_weight=1.0, clip_weight=2, 
                 strong_start_beams=20000,
                 strong_start_iters=6,
                 beam_add=768,
                 beam_sub=768//2,
                 min_beams=768,
                 max_len=75,
                 target_sim=1.0,
                 enable_upgrades=True,
                 ascii_only=True, allow_banned=False,
                 device=None, verbose=True):
        self.model_beams = model_beams
        self.clip_beams = clip_beams
        
        self.num_inter_beams = num_inter_beams
        self.max_inter_beams = max_inter_beams
        
        self.target_sim = target_sim

        self.enable_upgrades = enable_upgrades
        self.rating_weight = rating_weight
        self.clip_weight = clip_weight
        self.ascii_only = ascii_only
        self.allow_banned = allow_banned
        
        self.beam_add = beam_add
        self.beam_sub = beam_sub
        self.min_beams = min_beams
        
        self.max_len = max_len
        
        self.strong_start_beams = strong_start_beams
        self.strong_start_iters = strong_start_iters
        
        if device is None:
            device = torchutils.get_default_device()
        self.device = device
        
        self.verbose = verbose
        
class BeamSearcherFactory():
    def __init__(self, to_token_model, to_rating_model, clip_model):
        self._upgrade_config = UpgradeConfig()
        self._search_config = BeamSearchConfig()
        self._to_token_model = to_token_model
        self._to_rating_model = to_rating_model
        self._clip_model = clip_model
        
    def build(self):
        self._validate()
        return BeamSearcher(self._to_token_model, self._to_rating_model, self._clip_model,
                            copy.deepcopy(self._search_config), copy.deepcopy(self._upgrade_config))
        
    def _validate(self):
        if self._to_token_model is None:
            raise Exception("Tokens model not set")
        if self._to_rating_model is None:
            raise Exception("Ratings model not set")
        if self._clip_model is None:
            raise Exception("Clip model not set")
        
    def models(self, to_token_model, to_rating_model, clip_model):
        self._to_token_model = to_token_model
        self._to_rating_model = to_rating_model
        self._clip_model = clip_model
        return self
        
    def beams(self, model_beams, clip_beams):
        self._search_config.model_beams = model_beams
        self._search_config.clip_beams = clip_beams
        return self
    
    def no_improve_changes(self, beam_add, beam_sub):
        self._search_config.beam_add = beam_add
        self._search_config.beam_sub = beam_sub
        return self
    
    def interbeams(self, min_beams, num_inter_beams, max_inter_beams):
        self._search_config.min_beams = min_beams
        self._search_config.num_inter_beams = num_inter_beams
        self._search_config.max_inter_beams = max_inter_beams
        return self
        
    def strong_start_beams(self, strong_start_beams, strong_start_iters):
        self._search_config.strong_start_beams = strong_start_beams
        self._search_config.strong_start_iters = strong_start_iters
        return self
    
    def target_sim(self, target_sim):
        self._search_config.target_sim = target_sim
        return self
    
    def enable_upgrades(self, enable_upgrades):
        self._search_config.enable_upgrades = enable_upgrades
        return self
    
    def ascii_only(self, ascii_only):
        self._search_config.ascii_only = ascii_only
        return self
    
    def allow_banned(self, allow_banned):
        self._search_config.allow_banned = allow_banned
        return self
        
    def verbose_search(self, verbose):
        self._search_config.verbose = verbose
        return self
        
    def clip_weight(self, clip_weight):
        self._search_config.clip_weight = clip_weight
        return self
    
    def rating_weight(self, rating_weight):
        self._search_config.rating_weight = rating_weight
        return self
        
    def device(self, device):
        self._search_config.device = device
        return self
    
    def max_upgrade_cands(self, cands):
        self.baseline_cands = cands
        return self
    
    def upgrade_iter_start(self, start_iter):
        self._upgrade_config.upgrade_iter_start = start_iter
        return self
    
    def upgrade_threshold(self, upgrade_threshold):
        self._upgrade_config.upgrade_threshold = upgrade_threshold
        return self
    
    def upgrade_cap_margin(self, cap_margin):
        self._upgrade_config.cap_margin = cap_margin
        return self
    
    def verbose_upgrade(self, verbose):
        self._upgrade_config.verbose = verbose
        return self
        

class BeamSearcher:
    _epsilon = 1e-6
    class BeamSearchState:
        # Variables related to a given beam search to reduce excessive method parameters
        def __init__(self, features: torch.tensor, memory: torch.tensor, search_config: BeamSearchConfig, upgrade_config: UpgradeConfig, max_len: int, start_tokens: torch.Tensor = None):
            self.features = features
            self.memory = memory
            self.max_len = max_len
            torch_zero = torch.tensor([0.0], device=search_config.device)
            self.final_beam_tokens = []

            if start_tokens is None:
                torch_start = torch.tensor([sdconsts.sot_token], device=search_config.device, dtype=torch.long)
                self.curr_beams = [(torch_zero, torch_start)]
                self.iter_num = 0
            else:
                if isinstance(start_tokens, str):
                    # Don't include the end token
                    start_tokens = clip.tokenize(start_tokens)[0].cuda()

                self.final_beam_tokens.append(start_tokens)
                self.iter_num = (sdutils.find_end_idx(start_tokens)-1)
                start_tokens = start_tokens.clone().long()
                end_idx = sdutils.find_end_idx(start_tokens)
                if end_idx != -1:
                    start_tokens = start_tokens[:end_idx]
                    
                self.curr_beams = [(torch_zero, start_tokens)]
                
            self.total_beams = search_config.model_beams + search_config.clip_beams
            self.best_match_cosine = torch.tensor(0, device=search_config.device)
            self.iter_without_improvement = 0
            self.config = search_config
            self._upgrade_config = upgrade_config
            
            if search_config.strong_start_iters > 0 and self.iter_num < search_config.strong_start_iters:
                self.curr_inter_beams = search_config.strong_start_beams
            else:
                self.curr_inter_beams = search_config.num_inter_beams
                
            self.times_upgraded = 0
            self.last_upgrade_iter = 0
            self.last_start_upgrade = 0

            self.next_beam_scores = None
            
        def get_upgrade_cands(self):
            con_upgrades = (self.iter_without_improvement//self._upgrade_config.upgrade_threshold)
            return self._upgrade_config.baseline_cands * con_upgrades
            
        def should_upgrade(self):
            return self.iter_without_improvement != 0 and self.iter_without_improvement % self._upgrade_config.upgrade_threshold == 0

        def on_new_best(self, new_beam, cosine_sim):
            self.curr_inter_beams = max(int(self.curr_inter_beams-self.config.beam_sub), self.config.min_beams)
            self.iter_without_improvement = 0
            self.best_match_cosine = cosine_sim
            self.final_beam_tokens.append(new_beam)
            
        def on_no_improvement(self):
            self.curr_inter_beams = int(self.curr_inter_beams+self.config.beam_add)
            

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

            rem_tokens = max_size - ret.size(0)
            if rem_tokens > 0:
                # Add padding for the remaining size
                ret = torch.cat((ret, torch.zeros(rem_tokens, device=device, dtype=torch.long)))

            return ret
            
    def __init__(self, tokens_model: FeaturesToTokensAesModel, ratings_model: FeaturesToRatingModel, clip_model: ClipModel, search_config=None, upgrade_config=None):
        self._tokens_model = tokens_model
        self._ratings_model = ratings_model
        self._clip_model = clip_model
        self._upgrader = decoderpipeline.PromptUpgrader(self._tokens_model, clip_model, self._ratings_model, rating_weight=0.0, verbose=True)
        if search_config is None:
            search_config = BeamSearchConfig()
        self._search_config = search_config

        if upgrade_config is None:
            upgrade_config = UpgradeConfig()
        self._upgrade_config = upgrade_config
        
    def beam_search(self, features, max_len=75, topk=1, start_tokens=None, print_freq=1):

        self._tokens_model.train(False)
        self._ratings_model.train(False)

        self._upgrader.set_verbose(self._upgrade_config.verbose)


        with torch.no_grad():
            with torch.autocast(device_type="cuda"):
                # Setup the Inital search state
                features = self._process_orig_features(features)
                memory = self._tokens_model.features_to_memory(features).squeeze(0)
                search_state = BeamSearcher.BeamSearchState(features, memory, max_len=max_len, start_tokens=start_tokens,
                                                            search_config=self._search_config, upgrade_config=self._upgrade_config)
                print(f"Starting beam search to find the top {topk} prompts of length {search_state.max_len}. Printing every {print_freq} tokens")
                
                if start_tokens:
                    print(f"Start tokens are {start_tokens}")
                tokens_to_find = min(search_state.max_len, self._tokens_model._seq_len - 1)

                for _ in range(search_state.iter_num, tokens_to_find):
                    force_upgrade = (search_state.iter_num==tokens_to_find-1)
                    self._do_beam_search_step(search_state, force_upgrade=force_upgrade)

                    if search_state.iter_num % print_freq == 0:
                        self._print_search_state(search_state)
                        
                    search_state.iter_num += 1
                    
                    if search_state.best_match_cosine >= self._search_config.target_sim:
                        print(f"Found tokens with greater than target sim {search_state.best_match_cosine}")
                        break
                
                self._print_search_state(search_state)
                
                # Choose k tokens with the highest estimated score
                best_tokens, final_cosine_sim = self._get_final_tokens(search_state, topk)

                return best_tokens, final_cosine_sim, search_state.features
    
    def _process_orig_features(self, features):
        # Take a weighted sum of the features and normalize
        features = features.view(sdconsts.feature_width)
        features = (features/features.size(0)).unsqueeze(0)
        features_norm = features.norm(dim=-1, keepdim=True)
        features = features / features_norm

        return features

    def _do_beam_search_step(self, search_state: BeamSearchState, force_upgrade=False):
        # Narrow the search space to tokens that the model estimates are likely good matches
        candidates = self._get_candidate_beams(search_state)

        # Create new token tensors based on the candidates
        next_beam_tokens, next_beam_tokens_full = self._get_next_tokens(candidates)

        # Choose tokens based on their cosine similarity with the features
        top_clip_idxs = self._get_top_cosine_sim(search_state, next_beam_tokens_full)

        if self._search_config.rating_weight != 0:
            # Nudge the selection process toward tokens estimated to increase the quality of the result
            self._apply_rating_bonus(search_state, next_beam_tokens_full)
        
        # Choose tokens with high estimated scores and high cosine similarity with the featurs
        top_clip, top_beam = self._get_top_idxs(search_state, top_clip_idxs)
        
        search_state.curr_beams = []
        self._upgrade_top(search_state, candidates, top_clip_idxs, top_beam, next_beam_tokens_full, force_upgrade=force_upgrade)
        #candidates[top_clip_idxs[0]].prob = candidates[top_beam[0]].prob
        
        top_idxs = torch.unique(torch.cat((top_clip,top_beam))).cpu().tolist()
        # Append the choosen tokens to their parent beam for the next iteration
        for top_idx in top_idxs:
            prob = candidates[top_idx].prob
            new_beam = next_beam_tokens[top_idx]
            
            search_state.curr_beams.append((prob, new_beam))

    def _upgrade_top(self, search_state, candidates, top_clip_idxs, top_beam_idxs, next_beam_tokens_full, force_upgrade=False):
        top_clip_beams = next_beam_tokens_full[top_clip_idxs]
        top_beam, cosine_sim = self._clip_model.rank_similarity(search_state.features, top_clip_beams, top_count=1)
        top_beam = top_beam[0]
        
        cosine_sim = torch.tensor(cosine_sim[0], device= top_beam.device)
        
        if cosine_sim < search_state.best_match_cosine:
            search_state.iter_without_improvement += 1
            
        tokens_added = candidates[0].prev_tokens.size(0)+1
        upgrades_allowed = self._search_config.enable_upgrades and tokens_added >= self._upgrade_config.upgrade_iter_start 
        upgrade_triggered = force_upgrade or search_state.should_upgrade()
        if upgrades_allowed and upgrade_triggered:
            print(f"Upgrading  {self._tokens_model.decode(top_beam)}")
         
            beam_eot_idx = sdutils.find_end_idx(top_beam)
            new_beam = top_beam.clone()
            state = self._upgrader._create_state(search_state.features, new_beam, memory=search_state.memory)
            curr_cands = search_state.get_upgrade_cands()
            
            new_beam, _ = self._upgrader.single_upgrade_pass(search_state.features, new_beam, state=state,
                                                                num_candidates=self._upgrade_config.baseline_cands,
                                                                decay_factor=1.0,
                                                                start_idx=beam_eot_idx-self._upgrade_config.cap_margin)
            new_beam, _ = self._upgrader.single_upgrade_pass(search_state.features, new_beam, state=state,
                                                                num_candidates=self._upgrade_config.baseline_cands,
                                                                decay_factor=1.0,
                                                                start_idx=1, end_idx=1+self._upgrade_config.cap_margin,)
            
            for div in [8, 8, 4, 4, 2]:
                    new_beam, _ = self._upgrader.single_upgrade_pass(search_state.features, new_beam, state=state,
                                                                    num_candidates=curr_cands//div, decay_factor=1.0)
            
            search_state.curr_beams.append((candidates[top_beam_idxs[0]].prob, new_beam[:beam_eot_idx].clone()))
            top_beam = new_beam
        
            upgraded_features = self._clip_model.features_from_tokens(top_beam.view(1, -1))
            cosine_sim = self._clip_model.cosine_similarity(search_state.features, upgraded_features)[0]
            search_state.times_upgraded += 1
            
        if cosine_sim > search_state.best_match_cosine:
            search_state.on_new_best(top_beam, cosine_sim)
        elif search_state.curr_inter_beams <= self._search_config.max_inter_beams:
            search_state.on_no_improvement()

    def _get_candidate_beams(self, search_state):
        # Get the model's estimate for the next token's probability for each beam
        curr_tokens = torch.stack(tuple((x[1] for x in search_state.curr_beams)))
        
        token_probs = self._tokens_model.get_next_probs(search_state.memory, curr_tokens, ascii_only=self._search_config.ascii_only)
        token_probs = self._safe_log(torch.relu(token_probs))
        token_probs = token_probs / token_probs.norm(dim=-1, keepdim=True)

        # Add the log of the token's probablity to its parent's cumulative probability
        new_probs = tuple(
            search_state.curr_beams[idx][0] + token_probs[idx]
            for idx in range(len(search_state.curr_beams))
        )

        # Find next token choices with the highest cumulative probability as 1D indices
        next_probs = torch.cat(new_probs).view(-1)
        beams_to_use = search_state.curr_inter_beams
        if search_state.iter_num >= self._search_config.strong_start_iters:
            search_state.curr_inter_beams = self._search_config.num_inter_beams
            
        next_beam_scores, candidate_idxs = next_probs.topk(int(beams_to_use), dim=-1)
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

    def _get_next_tokens(self, candidates):
        # Create a list of tokens for each beams and a tensor of those tokens with an EOT token and padding
        next_beam_tokens = []
        next_beam_tokens_full = torch.zeros(
            len(candidates),
            77,
            device=self._search_config.device,
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

        if self._search_config.clip_weight != 0:
            clip_bonus = torch.softmax(next_beam_cosine_sim_aug, dim=-1)
            clip_bonus = self._safe_log(clip_bonus) * self._search_config.clip_weight
            search_state.next_beam_scores += clip_bonus

        return top_clip_idxs

    def _apply_rating_bonus(self, search_state, next_beam_tokens_full):
        beam_features = self._clip_model.features_from_tokens(next_beam_tokens_full)
        ratings = self._ratings_model(beam_features).view(-1)
        ratings_score = torch.softmax(ratings, dim=-1)
        aug_ratings_bonus = self._safe_log(ratings_score) * self._search_config.rating_weight

        search_state.next_beam_scores += aug_ratings_bonus

    def _get_top_idxs(self, search_state, top_clip_idxs):
        _, top_beam_idxs = search_state.next_beam_scores.topk(self._search_config.model_beams, dim=-1)
        
        unique_cnt = torch.unique(torch.cat((top_beam_idxs, top_clip_idxs[:self._search_config.clip_beams]))).size(0)
        
        overlap = self._search_config.clip_beams+self._search_config.model_beams-unique_cnt

        return top_clip_idxs[:self._search_config.clip_beams+overlap], top_beam_idxs

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
        
        """
        #Check for retokenization inconsistencies. This can occur when multiple token sequences can represent the same prompt
        orig = self._clip_model.cosine_similarity(search_state.features, [search_state.final_beam_tokens[-1]])[0]

        telephone_tokens = clip.tokenize(self._tokens_model.decode(search_state.final_beam_tokens[-1]))[0].cuda()
        telephone = self._clip_model.cosine_similarity(search_state.features, [telephone_tokens])[0]
        
        print(f"Cosine telephone {search_state.best_match_cosine: 0.4f} recalc {orig:0.4f} to {telephone: 0.4f}")
        print("Tokens pre", search_state.final_beam_tokens[-1])
        print("Tokens post", telephone_tokens)
        """
        print(f"Top beam has cosine sim {search_state.best_match_cosine: 0.3f} with estimated quality of {rating: 0.3f}:\n {self._tokens_model.decode(search_state.final_beam_tokens[-1])[0]}\n")


    def _safe_log(self, tensor):
        return torch.log(tensor + self._epsilon)
