import random

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
                    print(f"New best {self.curr_best_score} to {new_best}")
                    self.curr_best_score = self.tmp_best_score
                    self.best_tokens = self.tmp_best_tokens.clone()
                    prefix = "Current best"

                self._parent._print_result(self.target_features, self.tmp_best_tokens, self.tmp_best_score, result_prefix=prefix)

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
        print("Creating new state")
        target_features = target_features.view(1,-1).float()
        target_features /= target_features.norm(dim=-1, keepdim=True)
        memory = self._tokens_model.features_to_memory(target_features).squeeze(0)
        curr_best_score = self._calculator.score_tokens(target_features, tokens)[0].item()
        return PromptUpgrader.PromptState(self, target_features, memory, tokens, curr_best_score)

    def beam_upgrade(self, target_features, tokens, baseline_candidate_size=1024, num_beams=4, ascii_only=True):
        curr_beams = [tokens]
        beam_probs = [0]

        with torch.no_grad():
            target_features = target_features.view(1,-1).float()
            target_features /= target_features.norm(dim=-1, keepdim=True)

            curr_best_score_tokens = tokens
            curr_best_score = self._calculator.score_tokens(target_features, tokens)[0].item()
            self._print_result(target_features, tokens, curr_best_score)

            def update_tokens_if_better(new_tokens, new_best):
                nonlocal curr_best_score_tokens
                nonlocal curr_best_score
                is_improved = _is_improvement_eps(new_best, curr_best_score)
                if is_improved:
                    curr_best_score = new_best
                    curr_best_score_tokens = new_tokens
                    self._print_result(target_features, curr_best_score_tokens, curr_best_score)

                return is_improved

            memory = self._tokens_model.features_to_memory(target_features).squeeze(0)
            
            for token_idx in range(1, len(tokens)):
                if tokens[token_idx] == _eot_token: 
                    break
                print("Token ", token_idx)
                curr_tokens = torch.stack(tuple((x[:token_idx] for x in curr_beams)))
                token_probs = self._tokens_model.get_next_probs(memory, curr_tokens, ascii_only=ascii_only)
                token_probs = self._safe_log(token_probs)
                token_probs = token_probs.size(-1) * token_probs / token_probs.norm(dim=-1, keepdim=True)

                new_probs = tuple(
                    beam_probs[idx] + token_probs[idx]
                    for idx in range(len(beam_probs))
                )

                next_probs = torch.cat(new_probs).view(-1)
                next_probs, candidate_idxs  = next_probs.topk(baseline_candidate_size, dim=-1)
                candidate_idxs = torchutils.unravel_torch_idx(candidate_idxs, len(clip_tokenizer.encoder))
                def make_candidate_beam(prob, idx):
                    new_beam = curr_beams[idx[0]].clone()
                    new_beam[token_idx] = idx[1]
                    prob = beam_probs[idx[0]] + prob
                    return prob, new_beam

                candidates = [make_candidate_beam(prob, idx) for prob, idx in zip(next_probs, candidate_idxs)]
                top_idx, top_scores = self._calculator.arg_sort(target_features, [x[1] for x in candidates], top_count=num_beams)

                print("Update")
                update_tokens_if_better(candidates[0][1], top_scores[0])
                beam_probs = [candidates[idx.item()][0] for idx in top_idx]
                curr_beams = [candidates[idx.item()][1] for idx in top_idx]
                
                
            return curr_best_score_tokens, curr_best_score

    def upgrade(self, target_features, tokens, candidate_cnt=1024, large_pass_factor=2, max_iters=10, add_stride=5, max_tokens=77, state=None, ascii_only=True):
        large_candidate_cnt = candidate_cnt*large_pass_factor

        # TODO: Mask place holders when ranking tokens
        with torch.no_grad():
            if not state:
                state = self._create_state(target_features, tokens)
            self._print_result(state.target_features, tokens, state.curr_best_score, result_prefix="Initial prompt")

            print("Running initial refinement")
            for _ in range(max_iters):
                old_best = state.curr_best_score
                self.single_upgrade_pass(state.target_features, state.best_tokens, large_candidate_cnt, state=state, ascii_only=ascii_only)
                if not _is_improvement_eps(state.curr_best_score, old_best):
                    break

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
                self.add_tokens(state.target_features, state.best_tokens, add_stride, state=state)

                print(f"Upgrade cycle at {pass_cnt}")
                self.single_upgrade_pass(state.target_features, state.best_tokens, pass_cnt, state=state, ascii_only=ascii_only)
                
                prompt_end = state.get_end_idx()

            state.remove_tmp_state()
            self.remove_tokens(state.target_features, state.best_tokens, state=state)

            print(f"Finishing upgrade with a single large pass")
            self.single_upgrade_pass(state.target_features, state.best_tokens, large_candidate_cnt, state=state, ascii_only=ascii_only)

            return state.get_best()

    def single_upgrade_pass(self, target_features, tokens, num_candidates, ascii_only=True, state=None, start_idx=1, end_idx=-1):
        with torch.no_grad():
            if not state:
                state = self._create_state(target_features, tokens)

            new_tokens, new_best = self._run_upgrade_cycle(target_features, tokens, state.curr_best_score, num_candidates, state.memory,
                                                           ascii_only=ascii_only, start_idx=start_idx, end_idx=end_idx)
            state.update_tokens_if_better(new_tokens, new_best)

        return state.get_best()

    def remove_tokens(self, target_features, tokens, state=None):
        with torch.no_grad():
            if not state:
                state = self._create_state(target_features, tokens)

            print("Testing token removal")
            while True:
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
                if not state.update_tokens_if_better(top_tokens[0][0], top_sim[0].item()):
                    break
                print("Removed token")

            return state.get_best()

    def add_tokens(self, target_features, init_tokens, num_to_add, second_pass_freq=5, second_pass_cnt=2, fill_char='.', state=None,
                  new_token_candidates=quarter_tokens_cnt, refine_pass=quarter_tokens_cnt//2, ascii_only=True):
        if not state:
            state = self._create_state(target_features, init_tokens)
        prompt_end = state.get_end_idx()
        orig_end = prompt_end
        num_added = 0
        
        print(f"Adding {num_to_add} tokens")
        fill_token = clip_tokenizer.encoder[fill_char]
        while num_added < num_to_add and prompt_end < 76:
            tokens = state.tmp_best_tokens.clone()
            tokens[prompt_end] = fill_token
            prompt_end += 1
            tokens[prompt_end] = _eot_token
            curr_best_score = self._calculator.score_tokens(target_features, state.tmp_best_tokens)[0].item()

            # Allow the score to temporarly drop after adding a new token
            state.set_tmp_state(tokens, curr_best_score)
            self.single_upgrade_pass(target_features, tokens, new_token_candidates, state=state, ascii_only=ascii_only, start_idx=prompt_end-1)
            num_added += 1
            if num_added % second_pass_freq == 0:
                print("Refining added tokens")
                for _ in range(second_pass_cnt):
                    self.single_upgrade_pass(target_features, tokens, refine_pass, state=state, ascii_only=ascii_only, start_idx=orig_end)
            
        print("Finished adding tokens. Final refine pass")
        self.single_upgrade_pass(target_features, tokens, refine_pass, state=state, ascii_only=ascii_only, start_idx=orig_end)

        return state.get_best()

    def _run_upgrade_cycle(self, target_features, tokens, curr_best_score, num_cands, memory, ascii_only, print_freq=0, start_idx=1, end_idx=-1):
        if end_idx == -1:
            end_idx = tokens.size(0)-1
        for curr_end in range(start_idx, end_idx):
            if tokens[curr_end] == _eot_token: 
                break
        
            if print_freq != 0 and (curr_end-1) % print_freq == 0:
                print(f"Finished {curr_end} tokens with {num_cands} candidates per token")
                self._print_result(target_features, tokens, curr_best_score, True)

            to_test = [tokens.clone()]

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


            if _is_improvement_eps(top_sim[0], curr_best_score):
                tokens = top_tokens[0][0]
                curr_best_score = top_sim[0].item()

                #print(f"New best at token {curr_end} with {num_cands} candidates per token")
                #self._print_result(target_features, tokens, curr_best_score, True)
                if num_cands*2 >= 64:
                    num_cands = int(max(64, num_cands//2))



        return tokens, curr_best_score

    def _run_double_upgrade_cycle(self, target_features, tokens, curr_best_score, num_cands, memory, ascii_only=True, print_freq=0, second_cand_div=16, start_idx=1, end_idx=-1):
        if end_idx == -1:
            end_idx = tokens.size(0)-1

        for curr_end in range(start_idx, end_idx):
            if tokens[curr_end] == _eot_token: 
                break
        
            if print_freq != 0 and (curr_end-1) % print_freq == 0:
                print("Finished {curr_end} tokens with {num_cands} candidates per token")
                self._print_result(target_features, tokens, curr_best_score, True)

            to_test = [tokens.clone()]
            if num_cands >= 40000:
                candidate_tokens = torch.tensor([idx for idx in range(1, len(clip_tokenizer.encoder)-1) if self._tokens_model._ascii_mask[idx] != 0])
            else:
                replacement_probs = self._tokens_model.get_next_probs(memory, tokens[:curr_end].unsqueeze(0), ascii_only=ascii_only)[0]
                _, candidate_tokens = replacement_probs.topk(num_cands, dim=-1)

            for replacement_token in candidate_tokens:
                if replacement_token == _eot_token:
                    continue
                new_tensor = tokens.clone()
                new_tensor[curr_end] = replacement_token

                best_tokens, _ = self._run_upgrade_cycle(target_features, new_tensor, curr_best_score,
                                                         num_cands//second_cand_div, memory,
                                                         ascii_only, print_freq=print_freq, start_idx=curr_end+1, end_idx=curr_end+2)
                to_test.append(best_tokens)

            # TODO: Fix duplcate work
            test_stack = torch.stack(tuple(to_test))
            top_tokens, top_sim = self._calculator.rank(target_features, test_stack, 1)
            if _is_improvement_eps(top_sim[0], curr_best_score):
                tokens = top_tokens[0][0]
                curr_best_score = top_sim[0].item()

        return tokens, curr_best_score

    def set_verbose(verbose):
        self._verbose = verbose

    def _print_result(self, target_features, tokens, curr_best_score, result_prefix="Current best"):
        if not self._verbose:
            return
        top_features = self._clip_model.features_from_tokens(tokens, verbosity=0)
        top_sim = self._clip_model.cosine_similarity(target_features, top_features)[0]
        top_rating = self._rating_model(top_features)[0].item()
        print(f"{result_prefix} {curr_best_score}. Cosine Similarity {top_sim}. Rating {top_rating}:\n{self._tokens_model.decode(tokens)}\n{tokens}")
    
    def _safe_log(self, tensor):
        return torch.log(tensor + 1e-5)
    
    def get_blank_prompt(self, length=75, placeholder_lst=None):
        if placeholder_lst is None:
            placeholder_lst = self._placeholder_lst

        return ' '.join([random.choice(placeholder_lst) for _ in range(length)])

def _is_improvement_eps(new_val, prev_val, improve_eps=1e-4):
        return new_val - prev_val > improve_eps
