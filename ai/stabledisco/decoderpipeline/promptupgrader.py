import random
import re

from torch.nn.modules.activation import F

import ai.stabledisco.constants as sdconsts
import ai.stabledisco.utils as sdutils
import clip
import torch
from ai.stabledisco.clipmodel import ClipModel
from ai.stabledisco.decoderpipeline.promptmetrics import \
    CombinedClipRatingCalculator
from clip.clip import _tokenizer as clip_tokenizer

all_tokens_cnt = len(clip_tokenizer.encoder)
half_tokens_cnt = all_tokens_cnt//2
quarter_tokens_cnt = half_tokens_cnt//2

class PromptUpgrader:
    class PromptState:
        def __init__(self, parent, target_features, memory, tokens, curr_best_score):
            self._parent = parent
            self.target_features = target_features
            self.memory = memory
            
            tokens = sdutils.change_rev(tokens, True)[0]
            self.best_tokens = tokens
            self.curr_best_score = curr_best_score
            
            self.tmp_best_tokens = tokens
            self.tmp_best_score = curr_best_score

        def set_tmp_state(self, tmp_best_tokens, tmp_best_score):
            tmp_best_tokens = sdutils.change_rev(tmp_best_tokens, True)[0]
            self.tmp_best_tokens = tmp_best_tokens.clone()
            self.tmp_best_score = tmp_best_score

        def remove_tmp_state(self):
            self.tmp_best_tokens = self.best_tokens.clone()
            self.tmp_best_score = self.curr_best_score

        def get_end_idx(self):
            return sdutils.find_end_idx(self.tmp_best_tokens)[0]

        def get_best(self, rev_ret):
            to_ret = self.tmp_best_tokens
            if rev_ret:
                to_ret = sdutils.rev_tokens(self.tmp_best_tokens)[0]
            return to_ret, self.curr_best_score

        def update_tokens_if_better(self, new_tokens, new_best):
            is_improved = _is_improvement_eps(new_best, self.tmp_best_score)
            if is_improved:
                self.tmp_best_score = new_best
                self.tmp_best_tokens = new_tokens.clone()
                prefix = "Temp best"
                if _is_improvement_eps(new_best, self.curr_best_score):
                    self.curr_best_score = self.tmp_best_score
                    self.best_tokens = self.tmp_best_tokens.clone()
                    prefix = "Current best"

                self._parent._print_result(self.target_features, new_tokens, new_best, result_prefix=prefix)

            return is_improved

    def __init__(self, tokens_model: torch.nn.Module, clip_model: ClipModel, rating_model: torch.nn.Module, rating_weight=1.0, calculator=None, placeholder_lst=None, improve_eps=1e-7, ascii_only=True, no_banned=True, verbose=True):
        self._tokens_model = tokens_model
        self._clip_model = clip_model
        self._rating_model = rating_model
        self._improve_eps = improve_eps
        self._ascii_only = ascii_only
        self._no_banned = no_banned
        self._verbose = verbose

        if not placeholder_lst:
            placeholder_lst = ['.']
        self._placeholder_lst = placeholder_lst

        if calculator is None:
            calculator = CombinedClipRatingCalculator(self._clip_model, self._rating_model, rating_weight=rating_weight)
        self._calculator = calculator

    def _create_state(self, target_features, prompt):
        if self._verbose:
            print("Creating new state")
        target_features = target_features.view(1,-1).float()
        target_features /= target_features.norm(dim=-1, keepdim=True)
        if isinstance(prompt, str):
            tokens = clip.tokenize(prompt)[0].cuda()
        else:
            tokens = prompt.clone()
        memory = self._tokens_model.features_to_memory(target_features).squeeze(0)
        curr_best_score = self._calculator.score_tokens(target_features, tokens)[0].item()
        return PromptUpgrader.PromptState(self, target_features, memory, tokens, curr_best_score)

    def upgrade(self, target_features, prompt, candidate_cnt=4096, first_pass_mul=1.5, max_iters=6, add_stride=3, max_tokens=sdconsts.prompt_token_len, orig_start_idx=1, orig_end_idx=sdconsts.prompt_token_len, state=None):
        # TODO: Mask place holders when ranking tokens
        with torch.no_grad():
            if not state:
                state = self._create_state(target_features, prompt)
            tokens = state.best_tokens
            self._print_result(state.target_features, tokens, state.curr_best_score, result_prefix="Initial prompt")
            if self._verbose:
                print("Running initial refinement")

            step_size = 5
            cand_mul = 4
            max_muls = 2
            min_cands = int(candidate_cnt / (cand_mul ** max_muls))
            max_cands = candidate_cnt
            first_pass = int(candidate_cnt * first_pass_mul)
            
            max_tokens = min(sdconsts.prompt_token_len, max_tokens)
            self.single_upgrade_pass(state.target_features, state.best_tokens, min_cands, state=state, start_idx=1, rev_ret=False)
                        
            self.add_tokens(state.target_features, state.best_tokens, max_tokens=max_tokens, add_stride=add_stride, state=state, rev_ret=False)
            state.remove_tmp_state()
            
            for _ in range(3):
                for curr_start in range(orig_start_idx, min(state.get_end_idx() - step_size, orig_end_idx), step_size):
                    got_improvement = False
                    curr_cands = first_pass
                    for _ in range(max_iters//2):
                        old_best = state.curr_best_score
                        curr_end_idx=curr_start+step_size
                        if self._verbose:
                            print(f"Upgrade pass at {curr_cands} from {curr_start} to {curr_end_idx}")
                        self.single_upgrade_pass(state.target_features, state.best_tokens, curr_cands, state=state, start_idx=curr_start, end_idx=curr_end_idx, rev_ret=False)
                        got_improvement = _is_improvement_eps(state.curr_best_score, old_best)
                            
                    if not got_improvement:
                        if curr_cands >= max_cands:
                            continue
                        else:
                            curr_cands *= cand_mul
                    elif curr_cands > max_cands:
                        curr_cands = min_cands
                    elif curr_cands > min_cands:
                        curr_cands = max(min_cands, int(curr_cands / cand_mul))

            
            self.remove_tokens(state.target_features, state.best_tokens, state=state, start_idx=orig_start_idx, rev_ret=False)
            self.add_tokens(target_features, state.best_tokens, max_tokens=max_tokens, add_stride=add_stride, state=state, rev_ret=False)
            state.remove_tmp_state()
            
            if self._verbose:
                print(f"Finishing upgrade with a single large pass")
            self.single_upgrade_pass(state.target_features, state.best_tokens, first_pass, state=state, start_idx=orig_start_idx, rev_ret=False)

            return state.get_best(True)

    def single_upgrade_pass(self, target_features, prompt, num_candidates, decay_factor=0.9, state=None, start_idx=1, end_idx=-1, rev_ret=True):
        num_candidates = min(num_candidates, sdconsts.num_tokens-2)
        start_idx = max(1, start_idx)

        with torch.no_grad():
            with torch.autocast(device_type="cuda"):
                if not state:
                    state = self._create_state(target_features, prompt)
                
                tokens = state.tmp_best_tokens

                new_tokens, new_best = self._swap(target_features, tokens, state.tmp_best_score, start_idx=start_idx, end_idx=end_idx)
                state.update_tokens_if_better(new_tokens, new_best)

                new_tokens, new_best = self._run_upgrade_cycle(target_features, tokens, state.tmp_best_score, num_candidates, state.memory,
                                                            decay_factor=decay_factor, start_idx=start_idx, end_idx=end_idx)

                state.update_tokens_if_better(new_tokens, new_best)
                
                new_tokens, new_best = self._swap(target_features, tokens, state.tmp_best_score, start_idx=start_idx, end_idx=end_idx)
                state.update_tokens_if_better(new_tokens, new_best)

                return state.get_best(rev_ret=rev_ret)

    # TODO: Strangeness. It sometimes removes nothing with an apparent score increase
    def remove_tokens(self, target_features, prompt, state=None, max_remove = 5, start_idx=1, rev_ret=True):
        if self._verbose:
            print("Token removal disabled due to bugs")
        
        return state.get_best(rev_ret=rev_ret)
    
        with torch.no_grad():
            if not state:
                state = self._create_state(target_features, prompt)
                
            tokens = state.tmp_best_tokens
            if self._verbose:
                print("Testing token removal")
                
            for num_removed in range(max_remove):
                to_test = [tokens.clone()]
                for curr_end in range(start_idx, tokens.size(0)-1):
                    if tokens[curr_end] == sdconsts.sot_token: 
                        break

                    new_tensor = tokens.clone()
                    new_tensor = torch.cat((new_tensor[:curr_end], new_tensor[curr_end+1:], torch.tensor([0], device=new_tensor.device)))
                    to_test.append(new_tensor)

                for curr_end in range(1, tokens.size(0)-2):
                    if tokens[curr_end] == sdconsts.sot_token: 
                        break

                    new_tensor = tokens.clone()
                    new_tensor = torch.cat((new_tensor[:curr_end], new_tensor[curr_end+2:], torch.tensor([0, 0], device=new_tensor.device)))
                    to_test.append(new_tensor)

                test_stack = torch.stack(tuple(to_test))
                top_tokens, top_sim = self._calculator.rank(target_features, test_stack, 1)
                if not state.update_tokens_if_better(top_tokens[0][0], top_sim[0].item()):
                    break
                
                if self._verbose:
                    print(f"Removed {num_removed+1} tokens {self._tokens_model.decode(top_tokens[0][0])}")

            return state.get_best(rev_ret=rev_ret)
        
    def add_tokens(self, target_features, prompt, add_stride=3, max_tokens=sdconsts.prompt_token_len,
                   new_token_candidates=all_tokens_cnt//3, refine_mul = 2/3, final_mul=1/2, state=None, rev_ret=True):
        if not state:
            state = self._create_state(target_features, prompt)
        prompt_end = state.get_end_idx()
        
        rem_tokens = max_tokens - (prompt_end-1)
        while rem_tokens > 0:
            self._add_tokens(state, num_to_add=min(add_stride, rem_tokens), new_token_candidates=new_token_candidates, refine_mul=refine_mul, final_mul=final_mul)
            
            prompt_end = state.get_end_idx()
            if prompt_end >= max_tokens:
                break
        
        state.remove_tmp_state()
        
        return state.get_best(rev_ret)

    def _add_tokens(self, state, num_to_add, new_token_candidates, refine_mul, final_mul, refine_freq=3, refine_cnt=2):
        prompt_end = state.get_end_idx()
        orig_end = prompt_end
        num_added = 0
        if self._verbose:
            print(f"Adding {num_to_add} tokens")
            
        num_without_improve = 0
        max_without_improve = 2
        fill_token = 0
        refine_cands = int(new_token_candidates * refine_mul)
        final_pass = int(new_token_candidates * final_mul)
        while num_added < num_to_add and prompt_end < sdconsts.prompt_token_len-1:
            tokens = state.tmp_best_tokens.clone()
            tokens = sdutils.change_rev(tokens, False)[0]
            prompt_end = state.get_end_idx()
            tokens[prompt_end] = fill_token
            prompt_end += 1
            tokens[prompt_end] = sdconsts.eot_token
            tokens = sdutils.change_rev(tokens, True)[0]
            
            curr_best_score = self._calculator.score_tokens(state.target_features, tokens)[0].item()
            
            # Allow the score to temporarly drop after adding a new token
            state.set_tmp_state(tokens, curr_best_score)
            
            self.single_upgrade_pass(state.target_features, tokens, new_token_candidates, state=state, start_idx=max(1, prompt_end-2))
            if state.tmp_best_score <= curr_best_score:
                self.single_upgrade_pass(state.target_features, tokens, new_token_candidates, state=state, start_idx=max(1, orig_end-2))
                num_without_improve += 1
            else:
                num_without_improve = 0
                
            if num_without_improve >= max_without_improve:
                    break

            num_added += 1
            if (num_added) % refine_freq == 0:
                if self._verbose:
                    print(f"Refining added tokens at  {refine_cands}")
                for _ in range(refine_cnt):
                    self.single_upgrade_pass(state.target_features, tokens, refine_cands, state=state, start_idx=orig_end)
        
        if self._verbose:
            print(f"Finished adding tokens. Final refine pass at {final_pass}")
        self.single_upgrade_pass(state.target_features, tokens, final_pass, state=state, start_idx=orig_end)
        
        return state.get_best(False)

    def _swap(self, target_features, tokens, curr_best_score, start_idx=1, end_idx=-1, max_swaps=5):
        if end_idx == -1:
            end_idx = tokens.size(0)-1
        end_idx = min(end_idx, tokens.size(0)-1)
        swap_idxs = [(src, dst) for src in range(start_idx, end_idx) for dst in range(src, end_idx)]

        for _ in range(max_swaps):
            to_test = [tokens.clone()]
            for src, dst in swap_idxs:
                if src == dst or tokens[src] == sdconsts.sot_token or tokens[dst] == sdconsts.sot_token:
                    continue
                new_tensor = tokens.clone()
                new_tensor[src], new_tensor[dst] = new_tensor[dst], new_tensor[src]
                to_test.append(new_tensor)

            # TODO: Fix duplcate work
            test_stack = torch.stack(tuple(to_test))
            top_tokens, top_sim = self._calculator.rank(target_features, test_stack, 1)

            if _is_improvement_eps(top_sim[0], curr_best_score):
                tokens = top_tokens[0][0]
                curr_best_score = top_sim[0].item()
            else:
                break

        return tokens, curr_best_score


    def _run_upgrade_cycle(self, target_features, tokens, curr_best_score, num_cands, memory, decay_factor=1.0, print_freq=0, start_idx=1, end_idx=-1):
        start_idx = max(1, start_idx)
        # TODO: Wrap end idx
        last_idx = sdutils.find_end_idx(tokens)
        if end_idx == -1:
            end_idx = last_idx
            
        if end_idx > last_idx:
            to_upgrade = list(range(start_idx, last_idx)) + list(range(1, max(start_idx - last_idx, 2)))
        else:
            to_upgrade = list(range(start_idx, end_idx))

        to_upgrade = [last_idx - idx for idx in to_upgrade]
        ends = []
        for curr_end in to_upgrade:
            if tokens[curr_end] in {sdconsts.sot_token, sdconsts.eot_token}: 
                continue
        
            if self._verbose and print_freq != 0 and (curr_end-1) % print_freq == 0:
                print(f"Finished {curr_end} tokens with {num_cands} candidates per token")
                self._print_result(target_features, tokens, curr_best_score, True)

            

            replacement_probs = self._tokens_model.get_next_probs(memory, rev_tokens=tokens[:curr_end].unsqueeze(0), ascii_only=self._ascii_only, no_banned=self._no_banned)[0]
            _, candidate_tokens = replacement_probs.topk(num_cands, dim=-1)

            to_test = [tokens.clone()]

            for replacement_token in candidate_tokens:
                if replacement_token == sdconsts.sot_token:
                    continue
                
                new_tensor = tokens.clone()

                new_tensor[curr_end] = replacement_token
                to_test.append(new_tensor)

            # TODO: Fix duplcate work
            test_stack = torch.stack(tuple(to_test))
            top_tokens, top_sim = self._calculator.rank(target_features, test_stack, 1)

            if _is_improvement_eps(top_sim[0], curr_best_score):
                ends.append(curr_end)
                tokens = top_tokens[0][0]
                curr_best_score = top_sim[0].item()

                #print(f"New best at token {curr_end} with {num_cands} candidates per token")
                #self._print_result(target_features, tokens, curr_best_score, True)

                if decay_factor == 0:
                    break

                num_cands = int(max(128, int(num_cands*decay_factor)))
        #print(f"Replacements at {ends}")
        return tokens, curr_best_score

    def set_verbose(self, verbose):
        self._verbose = verbose

    def _print_result(self, target_features, tokens, curr_best_score, result_prefix="Current best"):
        if not self._verbose:
            return
        
        tokens = sdutils.change_rev(tokens, False)[0]
        top_features = self._clip_model.features_from_tokens(tokens, verbosity=0)
        top_sim = self._clip_model.cosine_similarity(target_features, top_features)[0]
        top_rating = self._rating_model(top_features)[0].item()
        print(f"{result_prefix} {curr_best_score}. Cosine Similarity {top_sim}. Rating {top_rating}:\n{self._tokens_model.decode(tokens)[0]}")
    
    def _safe_log(self, tensor):
        return torch.log(tensor + 1e-6)
    
    def get_blank_prompt(self, length=75, placeholder_lst=None):
        if placeholder_lst is None:
            placeholder_lst = self._placeholder_lst

        return ' '.join([random.choice(placeholder_lst) for _ in range(length)])

def _is_improvement_eps(new_val, prev_val, improve_eps=1e-4):
        return new_val - prev_val > improve_eps
