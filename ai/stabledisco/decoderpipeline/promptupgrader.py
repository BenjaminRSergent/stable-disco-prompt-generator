import random

import clip
import numpy as np
import torch
from ai.stabledisco.decoderpipeline.promptmetrics import \
    CombinedClipRatingCalculator
from clip.clip import _tokenizer as clip_tokenizer

_eot_token = clip_tokenizer.encoder["<|endoftext|>"]

class PromptUpgrader:
    def __init__(self, tokens_model, clip_model, rating_model, calculator=None, placeholder_lst=None, improve_eps=1e-4):
        self._tokens_model = tokens_model
        self._clip_model = clip_model
        self._rating_model = rating_model
        self._improve_eps = improve_eps

        if not placeholder_lst:
            placeholder_lst = ['.']
        self._placeholder_lst = placeholder_lst

        if calculator is None:
            calculator = CombinedClipRatingCalculator(self._clip_model, self._rating_model)
        self._calculator = calculator

    def upgrade(self, target_features, tokens, baseline_candidate_size=1024, max_iters=10, double_pass=False, ascii_only=True, verbose=True):
        small_pass_cnts = [baseline_candidate_size//4,
                           baseline_candidate_size//2]
        med_pass = baseline_candidate_size
        large_pass = baseline_candidate_size*2
        huge_pass = baseline_candidate_size*4

        # TODO: Mask place holders when ranking tokens
        # TODO: Reduce indentation
        with torch.no_grad():
            target_features = target_features.view(1,-1).float()
            target_features /= target_features.norm(dim=-1, keepdim=True)

            memory = self._tokens_model.features_to_memory(target_features).squeeze(0)

            curr_best = self._calculator.score_tokens(target_features, tokens)[0].item()
            prev_best = curr_best

            def update_tokens_if_better(new_tokens, new_best):
                nonlocal tokens
                nonlocal curr_best
                is_improved = self._is_improvement(new_best, curr_best)
                if is_improved:
                    curr_best = new_best
                    tokens = new_tokens
                    self._print_best(target_features, tokens, curr_best, verbose)

                return is_improved
            
            self._print_best(target_features, tokens, curr_best, verbose)

            max_tokens = len(clip_tokenizer.encoder)//2

            upgrade_cycle_func = self._run_double_upgrade_cycle if double_pass else self._run_upgrade_cycle

            tokens = clip.tokenize(".")[0].cuda()
            period_token = tokens[1]
            add_per = 2
            prompt_end = 2
            while prompt_end < 75:
                if (prompt_end-2) == 0:
                    pass_cnt = num_tokens
                elif (prompt_end-2) % 18 == 0:
                    pass_cnt = huge_pass
                else:
                    pass_cnt = large_pass

                got_improvement = True
                while got_improvement:
                    got_improvement = False
                    
                    new_tokens, new_best = upgrade_cycle_func(target_features, tokens, curr_best, pass_cnt, memory, ascii_only)
                    got_improvement = update_tokens_if_better(new_tokens, new_best) or got_improvement

                for _ in range(3):
                    print("Adding tokens")
                    prev_prompt_end = prompt_end
                    prompt_end += add_per
                    tokens[prev_prompt_end:prompt_end] = period_token
                    tokens[prompt_end] = _eot_token
                    new_tokens, new_best = upgrade_cycle_func(target_features, tokens, curr_best, num_tokens, memory, ascii_only, start_idx=prev_prompt_end, end_idx=prev_prompt_end+add_per)
                    if not update_tokens_if_better(new_tokens, new_best):
                        print("No improvement after adding tokens")
                        break
                    new_tokens, new_best = upgrade_cycle_func(target_features, tokens, curr_best, num_tokens, memory, ascii_only, start_idx=prev_prompt_end, end_idx=prev_prompt_end+add_per-1)
                    if not update_tokens_if_better(new_tokens, new_best):
                        continue
                    new_tokens, new_best = upgrade_cycle_func(target_features, tokens, curr_best, num_tokens, memory, ascii_only, start_idx=prev_prompt_end, end_idx=prev_prompt_end+add_per-2)
                    if not update_tokens_if_better(new_tokens, new_best):
                        continue

            while True:
                print("Testing token removal")
                to_test = [tokens.clone()]
                for curr_end in range(1, tokens.size(0)-1):
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
                if not update_tokens_if_better(top_tokens[0][0], top_sim[0].item()):
                    break
                print("Removed token")

            print(f"Finishing upgrade with a single large pass")
            new_tokens, new_best = upgrade_cycle_func(target_features, tokens, curr_best, huge_pass, memory, ascii_only)
            update_tokens_if_better(new_tokens, new_best)
            return tokens, curr_best

            print("Entering main upgrade loop")
            iter_num = 0
            last_iter_best = curr_best
            while iter_num < max_iters:
                iter_num += 1
                for num_cands in small_pass_cnts:
                    while self._is_improvement(curr_best, prev_best):
                        prev_best = curr_best
                        new_tokens, new_best = upgrade_cycle_func(target_features, tokens, curr_best, num_cands, memory, ascii_only)
                        # TODO: Fix duplication
                        update_tokens_if_better(new_tokens, new_best)

                new_tokens, new_best = upgrade_cycle_func(target_features, tokens, curr_best, large_pass, memory, ascii_only)
                if not self._is_improvement(new_best, curr_best):
                    new_tokens, new_best = upgrade_cycle_func(target_features, tokens, curr_best, huge_pass, memory, ascii_only)
                    if not self._is_improvement(new_best, curr_best):
                        return tokens, curr_best

                update_tokens_if_better(new_tokens, new_best)

                if self._is_improvement(curr_best, last_iter_best):
                    print("No improvement during iteration.")
                    break
                
                last_iter_best = curr_best

            print("Finished main update loop.")
            # Exit by breaking once there are no tokens that can be removed for an improvement
            while True:
                print("Testing token removal")
                to_test = [tokens.clone()]
                for curr_end in range(1, tokens.size(0)-1):
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
                if not update_tokens_if_better(top_tokens[0][0], top_sim[0].item()):
                    break
                print("Removed token")

            print(f"Finishing upgrade with a single large pass")
            new_tokens, new_best = upgrade_cycle_func(target_features, tokens, curr_best, huge_pass, memory, ascii_only)
            update_tokens_if_better(new_tokens, new_best)

            return tokens, curr_best

    def _run_upgrade_cycle(self, target_features, tokens, curr_best, num_cands, memory, ascii_only, print_freq=0, start_idx=1, end_idx=-1):
        if end_idx == -1:
            end_idx = tokens.size(0)-1
        for curr_end in range(start_idx, end_idx):
            if tokens[curr_end] == _eot_token: 
                break
        
            if print_freq != 0 and (curr_end-1) % print_freq == 0:
                print(f"Finished {curr_end} tokens with {num_cands} candidates per token")
                self._print_best(target_features, tokens, curr_best, True)

            to_test = [tokens.clone()]
            if num_cands >= 40000:
                candidate_tokens = torch.tensor(np.arange(1, len(clip_tokenizer.encoder)-1))
            else:
                replacement_probs = self._tokens_model.get_next_probs(memory, tokens[:curr_end].unsqueeze(0), ascii_only=ascii_only)[0]
                _, candidate_tokens = replacement_probs.topk(num_cands, dim=-1)
            for replacement_token in candidate_tokens:
                if replacement_token == _eot_token:
                    continue
                new_tensor = tokens.clone()

                new_tensor[curr_end] = replacement_token
                to_test.append(new_tensor)

            # TODO: Fix duplcate work
            test_stack = torch.stack(tuple(to_test))
            top_tokens, top_sim = self._calculator.rank(target_features, test_stack, 1)
            if self._is_improvement(top_sim[0], curr_best):
                tokens = top_tokens[0][0]
                curr_best = top_sim[0].item()

                print(f"New best at token {curr_end} with {num_cands} candidates per token")
                self._print_best(target_features, tokens, curr_best, True)
                if num_cands*2 >= 2048:
                    num_cands = int(max(1024, num_cands//2))


        return tokens, curr_best

    def _run_double_upgrade_cycle(self, target_features, tokens, curr_best, num_cands, memory, ascii_only, print_freq=0, second_cand_div=8):
        for curr_end in range(1, tokens.size(0)-2):
            if tokens[curr_end] == _eot_token: 
                break
        
            if print_freq != 0 and (curr_end-1) % print_freq == 0:
                print("Finished {curr_end} tokens with {num_cands} candidates per token")
                self._print_best(target_features, tokens, curr_best, True)

            to_test = [tokens.clone()]
            if num_cands >= 40000:
                candidate_tokens = torch.tensor(np.arange(1, len(clip_tokenizer.encoder)-1))
            else:
                replacement_probs = self._tokens_model.get_next_probs(memory, tokens[:curr_end].unsqueeze(0), ascii_only=ascii_only)[0]
                _, candidate_tokens = replacement_probs.topk(num_cands, dim=-1)

            for replacement_token in candidate_tokens:
                if replacement_token == _eot_token:
                    continue
                new_tensor = tokens.clone()
                new_tensor[curr_end] = replacement_token

                best_tokens, _ = self._run_upgrade_cycle(target_features, new_tensor, curr_best,
                                                         num_cands//second_cand_div, memory,
                                                         ascii_only, print_freq=print_freq, start_idx=curr_end+1, end_idx=curr_end+2)
                to_test.append(best_tokens)

            # TODO: Fix duplcate work
            test_stack = torch.stack(tuple(to_test))
            top_tokens, top_sim = self._calculator.rank(target_features, test_stack, 1)
            if self._is_improvement(top_sim[0], curr_best):
                tokens = top_tokens[0][0]
                curr_best = top_sim[0].item()

        return tokens, curr_best

    def _print_best(self, target_features, tokens, curr_best, verbose):
        if not verbose:
            return
        top_features = self._clip_model.features_from_tokens(tokens, verbosity=0)
        top_sim = self._clip_model.cosine_similarity(target_features, top_features)[0]
        top_rating = self._rating_model(top_features)[0].item()
        print(f"Current best {curr_best}. Cosine Similarity {top_sim}. Rating {top_rating}:\n{self._tokens_model.decode(tokens)}")

    def _is_improvement(self, new_val, prev_val):
        return new_val - prev_val > self._improve_eps
    
    def get_blank_prompt(self, length=75, placeholder_lst=None):
        if placeholder_lst is None:
            placeholder_lst = self._placeholder_lst

        return ' '.join([random.choice(placeholder_lst) for _ in range(length)])

    
