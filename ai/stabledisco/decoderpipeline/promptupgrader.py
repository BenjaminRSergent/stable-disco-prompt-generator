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
            
            self.passes_finished = 0

        def set_tmp_state(self, tmp_best_tokens, tmp_best_score):
            tmp_best_tokens = sdutils.change_rev(tmp_best_tokens, True)[0]
            self.tmp_best_tokens = tmp_best_tokens.clone()
            self.tmp_best_score = tmp_best_score

        def remove_tmp_state(self):
            self.tmp_best_tokens = self.best_tokens.clone()
            self.tmp_best_score = self.curr_best_score

        def get_end_idx(self):
            return sdutils.find_end_idx(self.tmp_best_tokens).item()

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
            tokens = clip.tokenize(prompt, truncate=True)[0].cuda()
        else:
            tokens = prompt.clone()
        memory = self._tokens_model.features_to_memory(target_features).squeeze(0)
        curr_best_score = self._calculator.score_tokens(target_features, tokens)[0].item()
        return PromptUpgrader.PromptState(self, target_features, memory, tokens, curr_best_score)

    # TODO: Token insertions
    def upgrade(self, target_features, prompt, candidate_cnt=4096, add_stride=10, max_tokens=sdconsts.prompt_token_len, orig_start_idx=1, orig_end_idx=sdconsts.prompt_token_len, max_iters=5, add_first = True, state=None):
        # TODO: Mask place holders when ranking tokens
        with torch.no_grad():
            if not state:
                state = self._create_state(target_features, prompt)
            tokens = state.best_tokens
            self._print_result(state.target_features, tokens, state.curr_best_score, result_prefix="Initial prompt")

            cand_mul = 1.5
            min_cands = int(candidate_cnt / cand_mul)
            max_cands = min(int(candidate_cnt*cand_mul), 5000)
            max_tokens = min(sdconsts.prompt_token_len, max_tokens)
            
            old_best = None
            new_best = state.curr_best_score
            
            self.remove_tokens(state.target_features, state.best_tokens, state=state, start_idx=orig_start_idx, rev_ret=False)
            
            end_idx = state.get_end_idx() 
            if add_first:
                self.add_tokens(state.target_features, state.best_tokens, pass_cands=min_cands, max_tokens=(max_tokens+end_idx)/2, add_stride=add_stride, state=state, rev_ret=False)
            
            early_divs = [8, 4, 2]
            for div in early_divs:
                _, new_best = self.single_upgrade_pass(state.target_features, state.best_tokens, min_cands//div, state=state, rev_ret=False)
            
            if self._verbose:
                print(f"Upgrading start and end tokens")
            
            
            _, new_best = self.single_upgrade_pass(state.target_features, state.best_tokens, max_cands, state=state, start_idx=end_idx-8, end_idx=end_idx, rev_ret=False, decay_factor=1.0)
            _, new_best = self.single_upgrade_pass(state.target_features, state.best_tokens, min_cands, state=state, start_idx=end_idx-16, end_idx=end_idx, rev_ret=False, decay_factor=1.0)
            print("Finished end pass")
              
            _, new_best = self.single_upgrade_pass(state.target_features, state.best_tokens, max_cands, state=state, start_idx=1, end_idx=8, rev_ret=False,decay_factor=1.0)
            _, new_best = self.single_upgrade_pass(state.target_features, state.best_tokens, min_cands, state=state, start_idx=1, end_idx=16, rev_ret=False, decay_factor=1.0)
            print("Finished start pass")
            
            if not add_first:
                self.add_tokens(state.target_features, state.best_tokens, pass_cands=min_cands, max_tokens=(max_tokens+end_idx)/2, add_stride=add_stride, state=state, rev_ret=False)
            
            
            end_idx = state.get_end_idx() 
            if self._verbose:
                print("Running initial refinement")
            for _ in range(max_iters):
                old_best = new_best
                # TODO: forward or backward heavy
                _, new_best = self.single_upgrade_pass(state.target_features, state.best_tokens, min_cands, state=state, start_idx=1, rev_ret=False)
                if not _is_improvement_eps(new_best, old_best):
                    break
                
            self.add_tokens(state.target_features, state.best_tokens, pass_cands=min_cands, max_tokens=max_tokens, add_stride=add_stride, state=state, rev_ret=False)
            state.remove_tmp_state()
            
            _, new_best = self.single_upgrade_pass(state.target_features, state.best_tokens, max_cands, state=state, rev_ret=False)
            old_best = new_best      
            self.remove_tokens(state.target_features, state.best_tokens, state=state, start_idx=orig_start_idx, rev_ret=False)
            if _is_improvement_eps(new_best, old_best): 
                self.add_tokens(target_features, state.best_tokens, max_tokens=max_tokens, add_stride=add_stride, state=state, rev_ret=False)
                state.remove_tmp_state()
                
            end_idx = state.get_end_idx() 
            
            if self._verbose:
                print(f"Finishing upgrade with a large pass followed by another refinement loop")
            self.single_upgrade_pass(state.target_features, state.best_tokens, num_candidates=max_cands, state=state, rev_ret=False)
            
            for _ in range(max_iters):
                old_best = new_best
                _, new_best = self.single_upgrade_pass(state.target_features, state.best_tokens, min_cands, state=state, rev_ret=False)
                if not _is_improvement_eps(new_best, old_best):
                    break
            
            if self._verbose:
                print(f"Upgrading end tokens with a large candidate count")
                  
            _, new_best = self.single_upgrade_pass(state.target_features, state.best_tokens, 5000, state=state, start_idx=end_idx-8, end_idx=end_idx, rev_ret=False, decay_factor=1.0)
            _, new_best = self.single_upgrade_pass(state.target_features, state.best_tokens, min_cands, state=state, start_idx=end_idx-16, end_idx=end_idx, rev_ret=False, decay_factor=1.0)
            print("Finished end pass")

            return state.get_best(True)

    def single_upgrade_pass(self, target_features, prompt, num_candidates, decay_factor=0.95, state=None, start_idx=1, end_idx=-1, rev_ret=True):
        num_candidates = min(num_candidates, sdconsts.num_tokens-2)
        start_idx = max(1, start_idx)

        with torch.no_grad():
            with torch.autocast(device_type="cuda"):
                if not state:
                    state = self._create_state(target_features, prompt)
                
                tokens = state.tmp_best_tokens

                new_tokens, new_best = self._swap(target_features, tokens, state.tmp_best_score, start_idx=start_idx, end_idx=end_idx)
                state.update_tokens_if_better(new_tokens, new_best)
                
                reverse_iterate = state.passes_finished % 2 == 0

                new_tokens, new_best = self._run_upgrade_cycle(target_features, new_tokens, new_best, num_candidates, state.memory,
                                                                            decay_factor=decay_factor, start_idx=start_idx, end_idx=end_idx, forward_weight=0, reverse_iterate = False)
                new_tokens, new_best = self._run_upgrade_cycle(target_features, new_tokens, new_best, num_candidates, state.memory,
                                                            decay_factor=decay_factor, start_idx=start_idx, end_idx=end_idx, forward_weight=None, reverse_iterate = reverse_iterate)
                new_tokens, new_best = self._run_upgrade_cycle(target_features, new_tokens, new_best, num_candidates, state.memory,
                                                                            decay_factor=decay_factor, start_idx=start_idx, end_idx=end_idx, forward_weight=1, reverse_iterate = True)

                state.update_tokens_if_better(new_tokens, new_best)

                state.passes_finished
                return state.get_best(rev_ret=rev_ret)

    # TODO: Strangeness. It sometimes removes nothing with an apparent score increase
    def remove_tokens(self, target_features, prompt, state=None, max_remove = 5, start_idx=1, rev_ret=True):
        with torch.no_grad():
            if not state:
                state = self._create_state(target_features, prompt)
                
            tokens = state.tmp_best_tokens
            if self._verbose:
                print("Testing token removal")
                
            for num_removed in range(max_remove):
                to_test = [tokens.clone()]
                
                for con_remove in range(1,10):
                    for curr_end in range(start_idx, state.get_end_idx()-con_remove):
                        if tokens[curr_end] == sdconsts.sot_token: 
                            break

                        new_tensor = tokens.clone()
                        new_tensor = torch.cat((new_tensor[:curr_end], new_tensor[curr_end+con_remove:], torch.zeros(con_remove, device=new_tensor.device, dtype=tokens.dtype)))
                        to_test.append(new_tensor)

                test_stack = torch.stack(tuple(to_test))
                top_tokens, top_sim = self._calculator.rank(target_features, test_stack, 1)
                if not state.update_tokens_if_better(top_tokens[0][0], top_sim[0].item()):
                    break
                
                if self._verbose:
                    print(f"Removed {num_removed+1} tokens {self._tokens_model.decode(top_tokens[0][0])}")

            return state.get_best(rev_ret=rev_ret)
        
    def add_tokens(self, target_features, tokens, max_insert_cands = 256, min_insert_cands=64, pass_cands=256, quick_pass_cands=32, add_stride=10, max_tokens=sdconsts.prompt_token_len,
                   new_token_candidates=3000, refine_mul = 5/3, final_mul=1, state=None, rev_ret=True):
        
        con_failed = 0
        max_con_failed = 5
        with torch.no_grad():
            if not state:
                state = self._create_state(target_features, tokens)
            
            rem_tokens = max_tokens - (state.get_end_idx()-1)
            
            if self._verbose:
                print(f"Adding {rem_tokens} tokens")
            for _ in range(int(rem_tokens)):
                percent_done = (state.get_end_idx())/max_tokens
                insert_cands = percent_done*min_insert_cands + (1-percent_done)*max_insert_cands
                
                old_best_score = state.tmp_best_score
                new_tokens, best_score = self._insert_token(state, num_cands=int(insert_cands))
                state.update_tokens_if_better(new_tokens, best_score)
                
                new_tokens, best_score = self.single_upgrade_pass(state.target_features, tokens, quick_pass_cands, state=state)
                state.update_tokens_if_better(new_tokens, best_score)
                
                if not _is_improvement_eps(old_best_score, state.curr_best_score):
                    new_tokens, best_score = self.single_upgrade_pass(state.target_features, new_tokens, pass_cands, state=state)
                    state.update_tokens_if_better(new_tokens, best_score)
                            
                if _is_improvement_eps(old_best_score, state.curr_best_score):
                    con_failed = 0
                else:
                    con_failed += 1
                    if con_failed >= max_con_failed:
                        break
            
            state.remove_tmp_state()
            return state.get_best(rev_ret)
        
    
    def _insert_token(self, state, num_cands=32*2, ave_to_check=15):
        if state.tmp_best_tokens[-1] == sdconsts.sot_token:
            print("Attempted to insert into a full token tensor")
            return
        
        prompt_end = state.get_end_idx()
        
        num_tokens = (prompt_end-1)
        percent_skip = 1 - (ave_to_check / max(num_tokens, 1))
        
        to_test = []
        to_upgrade = list(range(1, prompt_end+1))
        
        # Want 30
        # 10 = 3/1
        # 45 = 3/2
        
        for curr_end in to_upgrade:
            tokens = state.tmp_best_tokens.clone()
            if tokens[curr_end] == sdconsts.eot_token or random.random() < percent_skip: 
                continue
            
            tokens[curr_end+1:] = tokens[curr_end:-1].clone()
            tokens[curr_end] = 0
            curr_forward_weight = curr_end/prompt_end
            
            curr_end_idx = sdutils.find_end_idx(tokens)
            rev_tokens = sdutils.rev_tokens(tokens)[0]
            
            rev_idx = max(curr_end_idx-curr_end, 1)
            replacement_probs = self._tokens_model.get_next_probs(state.memory,
                                                                  forward_weight=curr_forward_weight,
                                                                  tokens=rev_tokens[:rev_idx].unsqueeze(0),
                                                                  rev_tokens=tokens[:curr_end].unsqueeze(0),
                                                                  ascii_only=self._ascii_only, no_banned=self._no_banned)[0]
            _, candidate_tokens = replacement_probs.topk(num_cands, dim=-1)

            for replacement_token in candidate_tokens:
                if replacement_token == sdconsts.sot_token:
                    continue
                
                new_tensor = tokens.clone()

                new_tensor[curr_end] = replacement_token
                to_test.append(new_tensor)

        # TODO: Fix duplcate work
        
        test_stack = torch.stack(tuple(to_test))
        top_tokens, top_sim = self._calculator.rank(state.target_features, test_stack, 1)
        state.set_tmp_state(top_tokens[0][0], top_sim[0].item())
        
        return top_tokens[0][0], top_sim[0].item()

    def _add_tokens(self, state, num_to_add, new_token_candidates, refine_mul, final_mul, pass_cands, refine_freq=5, refine_cnt=1):
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
        refine_extra_toks=2
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
            
            self.single_upgrade_pass(state.target_features, tokens, new_token_candidates, state=state, start_idx=prompt_end-refine_extra_toks)
            if state.tmp_best_score <= curr_best_score:
                self.single_upgrade_pass(state.target_features, tokens, new_token_candidates, state=state, start_idx=orig_end-refine_extra_toks)
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
                    self.single_upgrade_pass(state.target_features, tokens, refine_cands, state=state, start_idx=orig_end-refine_extra_toks)
        
        if self._verbose:
            print(f"Finished adding tokens. Final refine pass at {final_pass}")
        self.single_upgrade_pass(state.target_features, tokens, final_pass, state=state, end_idx=num_added)
        
        self.single_upgrade_pass(state.target_features, tokens, pass_cands, state=state)
        
        return state.get_best(False)

    def _swap(self, target_features, tokens, curr_best_score, start_idx=1, end_idx=-1, max_swaps=10):
        tokens_end = sdutils.find_end_idx(tokens)
        if end_idx == -1:
            end_idx = tokens_end
        end_idx = min(end_idx, tokens_end)
        
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

    def _run_upgrade_cycle(self, target_features, tokens, curr_best_score, num_cands, memory, ave_tokens=15, decay_factor=1.0, print_freq=0, start_idx=1, end_idx=-1, forward_weight=None, reverse_iterate=None):
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

        if reverse_iterate is None:
            reverse_iterate = forward_weight is not None and forward_weight > 0.5
        if reverse_iterate:
            to_upgrade = to_upgrade[::-1]
        
        orig_to_upgrade = list(to_upgrade)
        impact = sdutils.rank_token_impact(tokens, self._clip_model, idxs=orig_to_upgrade, target_features=target_features)
        to_upgrade = [(impact_idx, arg[1]) for impact_idx, arg in enumerate(impact[::-1])]

        ends = []
        total_iters = 0
        iter_num = 0
        num_checks = 0
        
        # at 10 = 35 * (1 - 35/10) = -87.5
        # at 35 = 35 * (1 - 35/35) = 0
        # at 70 = 35 * (1 - 35/70) = 17.5
        # at 1000 = 35 * (1 - 35/1000) = 33.775
        weight_mul = ave_tokens*(1 - (ave_tokens / max(last_idx,1 )))
        
        
        
        while iter_num < len(to_upgrade):
            impact_idx, curr_end = to_upgrade[iter_num]
            total_iters += 1
            iter_num += 1
            
            # Don't skip ave_tokens on average prefering low impact tokens 
            choice_chance = impact[impact_idx].impact*weight_mul
            skip = choice_chance < random.random()
            if skip or tokens[curr_end] in {sdconsts.sot_token, sdconsts.eot_token}: 
                continue
            num_checks += 1
            if self._verbose and print_freq != 0 and (curr_end-1) % print_freq == 0:
                print(f"Finished {curr_end} tokens with {num_cands} candidates per token")
                self._print_result(target_features, tokens, curr_best_score, True)

            if forward_weight is None:
                curr_forward_weight = curr_end/last_idx
            else:
                curr_forward_weight = forward_weight
                
            rev_tokens = sdutils.rev_tokens(tokens)[0]
            rev_idx = max(end_idx-curr_end, 1)
            replacement_probs = self._tokens_model.get_next_probs(memory,
                                                                  forward_weight=curr_forward_weight,
                                                                  tokens=rev_tokens[:rev_idx].unsqueeze(0),
                                                                  rev_tokens=tokens[:curr_end].unsqueeze(0),
                                                                  ascii_only=self._ascii_only,
                                                                  no_banned=self._no_banned)[0]
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
                
                #print(iter_num, len(to_upgrade))
                
                impact = sdutils.rank_token_impact(tokens, self._clip_model, idxs=orig_to_upgrade, target_features=target_features)
                
                rem = len(to_upgrade) - iter_num
                to_upgrade = [(impact_idx, arg[1]) for impact_idx, arg in enumerate(impact[:rem:-1])]
                iter_num = 0
                #print(f"New best at token {curr_end} with {num_cands} candidates per token {len(to_upgrade)}")
                #self._print_result(target_features, tokens, curr_best_score, True)

                if decay_factor == 0:
                    break

                num_cands = int(max(128, int(num_cands*decay_factor)))
     
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
        print(f"{result_prefix} {curr_best_score}. Cosine Similarity {top_sim}. Rating {top_rating}:\n{self._tokens_model.decode(tokens)[0]}\n")
        
    def _safe_log(self, tensor):
        return torch.log(tensor + 1e-6)
    
    def get_blank_prompt(self, length=75, placeholder_lst=None):
        if placeholder_lst is None:
            placeholder_lst = self._placeholder_lst

        return ' '.join([random.choice(placeholder_lst) for _ in range(length)])

def _is_improvement_eps(new_val, prev_val, improve_eps=1e-4):
        return new_val - prev_val > improve_eps
