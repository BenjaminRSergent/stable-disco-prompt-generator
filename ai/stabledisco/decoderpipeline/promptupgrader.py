import random
import re

import ai.torchmodules.utils as torchutils
import clip
import torch
from ai.stabledisco.decoderpipeline.promptmetrics import \
    CombinedClipRatingCalculator
from clip.clip import _tokenizer as clip_tokenizer

_eot_token = clip_tokenizer.encoder["<|endoftext|>"]

all_tokens_cnt = len(clip_tokenizer.encoder)
half_tokens_cnt = all_tokens_cnt//2
quarter_tokens_cnt = half_tokens_cnt//2

class PromptUpgrader:
    class PromptState:
        def __init__(self, parent, target_features, memory, tokens, curr_best_score):
            self._parent = parent
            self.target_features = target_features
            self.memory = memory
            
            self.best_tokens = tokens
            self.curr_best_score = curr_best_score
            
            self.tmp_best_tokens = tokens
            self.tmp_best_score = curr_best_score

        def set_tmp_state(self, tmp_best_tokens, tmp_best_score):
            self.tmp_best_tokens = tmp_best_tokens.clone()
            self.tmp_best_score = tmp_best_score

        def remove_tmp_state(self):
            self.tmp_best_tokens = self.best_tokens.clone()
            self.tmp_best_score = self.curr_best_score

        def get_end_idx(self):
            return torch.argwhere(self.tmp_best_tokens == _eot_token).view(-1)[0]

        def get_best(self):
            return self.best_tokens, self.curr_best_score

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

    def __init__(self, tokens_model, clip_model, rating_model, calculator=None, placeholder_lst=None, improve_eps=1e-7, verbose=True):
        self._tokens_model = tokens_model
        self._clip_model = clip_model
        self._rating_model = rating_model
        self._improve_eps = improve_eps
        self._verbose = verbose

        if not placeholder_lst:
            placeholder_lst = ['.']
        self._placeholder_lst = placeholder_lst

        if calculator is None:
            calculator = CombinedClipRatingCalculator(self._clip_model, self._rating_model)
        self._calculator = calculator

    def _create_state(self, target_features, tokens):
        if self._verbose:
            print("Creating new state")
        target_features = target_features.view(1,-1).float()
        target_features /= target_features.norm(dim=-1, keepdim=True)
        memory = self._tokens_model.features_to_memory(target_features).squeeze(0)
        curr_best_score = self._calculator.score_tokens(target_features, tokens)[0].item()
        return PromptUpgrader.PromptState(self, target_features, memory, tokens, curr_best_score)

    def upgrade(self, target_features, tokens, candidate_cnt=1024, large_pass_factor=2, max_init_iter=6, add_stride=5, max_tokens=77, start_idx=1, state=None, ascii_only=True):
        large_candidate_cnt = candidate_cnt*large_pass_factor

        # TODO: Mask place holders when ranking tokens
        with torch.no_grad():
            if not state:
                state = self._create_state(target_features, tokens)
            self._print_result(state.target_features, tokens, state.curr_best_score, result_prefix="Initial prompt")
            if self._verbose:
                print("Running initial refinement")

            step_size = 10
            refine_cnt = candidate_cnt
            for _ in range(3):
                for curr_start in range(start_idx, state.get_end_idx() - step_size, step_size):
                    got_improvement = False
                    for _ in range(max_init_iter//2):
                        old_best = state.curr_best_score
                        self.single_upgrade_pass(state.target_features, state.best_tokens, refine_cnt, state=state, ascii_only=ascii_only, start_idx=curr_start, end_idx=curr_start+step_size)
                        if _is_improvement_eps(state.curr_best_score, old_best):
                            got_improvement = True
                            
                    if not got_improvement:
                        if refine_cnt == large_candidate_cnt:
                                    break
                        else:
                            refine_cnt = large_candidate_cnt

            prompt_end = state.get_end_idx()
            num_cycles = 0
            
            while prompt_end < max_tokens-1:
                # There are often improvements smaller passes catch after large passes. Alternating medium and large passes has similar
                # quality results and saves time.
                if (num_cycles) % 2 == 0:
                    pass_cnt = candidate_cnt 
                else:
                    pass_cnt = large_candidate_cnt

                num_cycles += 1
                old_score = state.curr_best_score
                self.add_tokens(state.target_features, state.best_tokens, add_stride, state=state)
                if self._verbose:
                    print(f"Upgrade cycle at {pass_cnt}")
                self.single_upgrade_pass(state.target_features, state.best_tokens, pass_cnt, state=state, ascii_only=ascii_only, start_idx=start_idx)

                if not _is_improvement_eps(state.curr_best_score, old_score):
                    if self._verbose:
                        print("Failed to improve adding tokens, exiting add-token loop")
                        break
                
                prompt_end = state.get_end_idx()

            state.remove_tmp_state()
            self.remove_tokens(state.target_features, state.best_tokens, state=state, start_idx=start_idx)
            if self._verbose:
                print(f"Finishing upgrade with a single large pass")
            self.single_upgrade_pass(state.target_features, state.best_tokens, large_candidate_cnt, state=state, ascii_only=ascii_only, start_idx=start_idx)

            return state.get_best()

    def single_upgrade_pass(self, target_features, tokens, num_candidates, decay_factor=0.9, ascii_only=True, state=None, start_idx=1, end_idx=-1):
        with torch.no_grad():
            if not state:
                state = self._create_state(target_features, tokens)

            new_tokens, new_best = self._swap(target_features, tokens, state.tmp_best_score, start_idx=start_idx, end_idx=end_idx)
            state.update_tokens_if_better(new_tokens, new_best)

            new_tokens, new_best = self._run_upgrade_cycle(target_features, tokens, state.tmp_best_score, num_candidates, state.memory,
                                                           decay_factor=decay_factor, ascii_only=ascii_only, start_idx=start_idx, end_idx=end_idx)

            state.update_tokens_if_better(new_tokens, new_best)
            
            new_tokens, new_best = self._swap(target_features, tokens, state.tmp_best_score, start_idx=start_idx, end_idx=end_idx)
            state.update_tokens_if_better(new_tokens, new_best)

        return state.get_best()

    def remove_tokens(self, target_features, tokens, state=None, start_idx=1):
        with torch.no_grad():
            if not state:
                state = self._create_state(target_features, tokens)
            if self._verbose:
                print("Testing token removal")
            while True:
                to_test = [tokens.clone()]
                for curr_end in range(start_idx, tokens.size(0)-1):
                    if tokens[curr_end] == _eot_token: 
                        break

                    new_tensor = tokens.clone()
                    new_tensor = torch.cat((new_tensor[:curr_end], new_tensor[curr_end+1:], torch.tensor([0], device=new_tensor.device)))
                    to_test.append(new_tensor)

                for curr_end in range(1, tokens.size(0)-2):
                    if tokens[curr_end] == _eot_token: 
                        break

                    new_tensor = tokens.clone()
                    new_tensor = torch.cat((new_tensor[:curr_end], new_tensor[curr_end+2:], torch.tensor([0, 0], device=new_tensor.device)))
                    to_test.append(new_tensor)

                test_stack = torch.stack(tuple(to_test))
                top_tokens, top_sim = self._calculator.rank(target_features, test_stack, 1)
                if not state.update_tokens_if_better(top_tokens[0][0], top_sim[0].item()):
                    break
                if self._verbose:
                    print(f"Removed token {self._tokens_model.decode(top_tokens[0][0])}")

            return state.get_best()

    def add_tokens(self, target_features, init_tokens, num_to_add, second_pass_freq=2, second_pass_cnt=2, state=None,
                  new_token_candidates=(2*all_tokens_cnt)//3, refine_pass=quarter_tokens_cnt, final_pass=quarter_tokens_cnt//2, ascii_only=True):
        if not state:
            state = self._create_state(target_features, init_tokens)
        prompt_end = state.get_end_idx()
        orig_end = prompt_end
        num_added = 0
        if self._verbose:
            print(f"Adding {num_to_add} tokens")
        num_without_improve = 0
        max_without_improve = 2
        fill_token = 0
        while num_added < num_to_add and prompt_end < 76:
            tokens = state.tmp_best_tokens.clone()
            tokens[prompt_end] = fill_token
            prompt_end += 1
            tokens[prompt_end] = _eot_token
            curr_best_score = self._calculator.score_tokens(target_features, tokens)[0].item()

            # Allow the score to temporarly drop after adding a new token
            state.set_tmp_state(tokens, curr_best_score)
            
            self.single_upgrade_pass(target_features, tokens, new_token_candidates, state=state, ascii_only=ascii_only, start_idx=prompt_end-2)
            if state.tmp_best_score == curr_best_score:
                print(f"No good token diff {state.tmp_best_score } vs {curr_best_score}, trying to upgrade")
                self.single_upgrade_pass(target_features, tokens, new_token_candidates, state=state, ascii_only=ascii_only, start_idx=orig_end)
                
            if state.tmp_best_score == curr_best_score:
                num_without_improve += 1
                if num_without_improve >= max_without_improve:
                    break
            else:
                num_without_improve = 0

            num_added += 1
            if (num_added+1) % second_pass_freq == 0:
                if self._verbose:
                    print(f"Refining added tokens at  {refine_pass}")
                for _ in range(second_pass_cnt):
                    self.single_upgrade_pass(target_features, tokens, refine_pass, state=state, ascii_only=ascii_only, start_idx=orig_end)
        
        if self._verbose:
            print(f"Finished adding tokens. Final refine pass at {final_pass}")
        self.single_upgrade_pass(target_features, tokens, final_pass, state=state, ascii_only=ascii_only, start_idx=orig_end)

        return state.get_best()

    def _swap(self, target_features, tokens, curr_best_score, start_idx=1, end_idx=-1, max_swaps=5):
        if end_idx == -1:
            end_idx = tokens.size(0)-1
        end_idx = min(end_idx, tokens.size(0)-1)
        swap_idxs = [(src, dst) for src in range(start_idx, end_idx) for dst in range(src, end_idx)]

        for _ in range(max_swaps):
            to_test = [tokens.clone()]
            for src, dst in swap_idxs:
                if src == dst or tokens[src] == _eot_token or tokens[dst] == _eot_token:
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


    def _run_upgrade_cycle(self, target_features, tokens, curr_best_score, num_cands, memory, ascii_only,  decay_factor=1.0, print_freq=0, start_idx=1, end_idx=-1):
        if end_idx == -1:
            end_idx = tokens.size(0)-1
        end_idx = min(end_idx, tokens.size(0)-1)

        ends = []
        for curr_end in range(start_idx, end_idx):
            if tokens[curr_end] == _eot_token: 
                break
        
            if self._verbose and print_freq != 0 and (curr_end-1) % print_freq == 0:
                print(f"Finished {curr_end} tokens with {num_cands} candidates per token")
                self._print_result(target_features, tokens, curr_best_score, True)

            

            replacement_probs = self._tokens_model.get_next_probs(memory, tokens[:curr_end].unsqueeze(0), ascii_only=ascii_only)[0]
            _, candidate_tokens = replacement_probs.topk(num_cands, dim=-1)

            to_test = [tokens.clone()]

            
            for replacement_token in candidate_tokens:
                if replacement_token == _eot_token:
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
        print(f"Replacements at {ends}")
        return tokens, curr_best_score

    def set_verbose(self, verbose):
        self._verbose = verbose

    def _print_result(self, target_features, tokens, curr_best_score, result_prefix="Current best"):
        if not self._verbose:
            return
        top_features = self._clip_model.features_from_tokens(tokens, verbosity=0)
        top_sim = self._clip_model.cosine_similarity(target_features, top_features)[0]
        top_rating = self._rating_model(top_features)[0].item()
        print(f"{result_prefix} {curr_best_score}. Cosine Similarity {top_sim}. Rating {top_rating}:\n{self._tokens_model.decode(tokens)}")
    
    def _safe_log(self, tensor):
        return torch.log(tensor + 1e-6)
    
    def get_blank_prompt(self, length=75, placeholder_lst=None):
        if placeholder_lst is None:
            placeholder_lst = self._placeholder_lst

        return ' '.join([random.choice(placeholder_lst) for _ in range(length)])

def _is_improvement_eps(new_val, prev_val, improve_eps=1e-4):
        return new_val - prev_val > improve_eps
