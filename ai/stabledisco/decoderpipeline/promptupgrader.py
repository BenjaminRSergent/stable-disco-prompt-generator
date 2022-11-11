import copy
import random

import ai.stabledisco.constants as sdconsts
import ai.stabledisco.utils as sdutils
import clip
import torch
from ai.stabledisco.clipmodel import ClipModel
from ai.stabledisco.decoderpipeline.promptmetrics import CombinedClipRatingCalculator
from clip.clip import _tokenizer as clip_tokenizer

all_tokens_cnt = len(clip_tokenizer.encoder)
half_tokens_cnt = all_tokens_cnt // 2
quarter_tokens_cnt = half_tokens_cnt // 2


class PromptUpgraderFactory:
    def __init__(self, to_token_model, to_rating_model, clip_model):
        self._config = PromptUpgraderConfig()
        self._to_token_model = to_token_model
        self._to_rating_model = to_rating_model
        self._clip_model = clip_model

    def build(self):
        self._validate()
        return PromptUpgrader(
            self._to_token_model,
            self._to_rating_model,
            self._clip_model,
            config=copy.deepcopy(self._config),
        )

    def _validate(self):
        if self._to_token_model is None:
            raise Exception("Tokens model not set")
        if self._to_rating_model is None:
            raise Exception("Ratings model not set")
        if self._clip_model is None:
            raise Exception("Clip model not set")

    def max_cands(self, max_cands):
        self._config.max_cands = max_cands
        return self

    def cand_mul(self, cand_mul):
        self._config.cand_mul = cand_mul
        return self

    def rating_weight(self, rating_weight):
        self._config.rating_weight = rating_weight
        return self

    def max_token_removal(self, max_token_removal):
        self._config.max_token_removal = max_token_removal
        return self

    def removal_sim_thresh(self, removal_sim_thresh):
        self._config.removal_sim_thresh = removal_sim_thresh
        return self

    def calculator(self, calculator):
        self._config.calculator = calculator
        return self

    def improve_eps(self, improve_eps):
        self._config.improve_eps = improve_eps
        return self

    def num_insert_checks(self, num_insert_checks):
        self._config.num_insert_checks = num_insert_checks
        return self

    def quick_pass_cands(self, quick_pass_cands):
        self._config.quick_pass_cands = quick_pass_cands
        return self

    def insert_cands(self, insert_cands):
        self._config.insert_cands = insert_cands
        return self

    def max_iters(self, max_iters):
        self._config.max_iters = max_iters
        return self

    def replacement_iters(self, replacement_iters):
        self._config.replacement_iters = replacement_iters
        return self

    def add_first(self, add_first):
        self._config.add_first = add_first
        return self

    def do_large_cap_pass(self, do_large_cap_pass):
        self._config.do_large_cap_pass = do_large_cap_pass
        return self

    def ascii_only(self, ascii_only):
        self._config.ascii_only = ascii_only
        return self

    def no_banned(self, no_banned):
        self._config.no_banned = no_banned
        return self

    def verbose(self, verbose):
        self._config.verbose = verbose
        return self

    def target_rating(self, target_rating):
        self._config.target_rating = target_rating
        return self

    def target_sim(self, target_sim):
        self._config.target_sim = target_sim
        return self


class PromptUpgraderConfig:
    # Default values are based on a bayesian optimization parameter search
    def __init__(
        self,
        max_cands=5000,
        cand_mul=2,
        max_iters=15,
        replacement_iters=2,
        insert_cands=1024 * 2,
        rating_weight=1.0,
        max_token_removal=5,
        removal_sim_thresh=0.1,
        calculator=None,
        improve_eps=1e-7,
        num_insert_checks=20,
        quick_pass_cands=None,
        target_rating=10.0,
        target_sim=1.0,
        add_first=True,
        do_large_cap_pass=True,
        ascii_only=True,
        no_banned=True,
        verbose=True,
    ):
        if quick_pass_cands is None:
            quick_pass_cands = [2, 4, 8, 16]

        self.max_cands = max_cands
        self.cand_mul = cand_mul
        self.max_iters = max_iters
        self.replacement_iters = replacement_iters
        self.insert_cands = insert_cands
        self.num_insert_checks = num_insert_checks
        self.quick_pass_cands = quick_pass_cands

        self.rating_weight = rating_weight
        self.max_token_removal = max_token_removal
        self.removal_sim_thresh = removal_sim_thresh

        self.calculator = calculator
        self.improve_eps = improve_eps

        self.target_rating = target_rating
        self.target_sim = target_sim

        self.add_first = add_first
        self.do_large_cap_pass = do_large_cap_pass
        self.ascii_only = ascii_only
        self.no_banned = no_banned
        self.verbose = verbose


class PromptUpgrader:
    class PromptState:
        class BatchContext:
            def __init__(self, state):
                self._state = state
                self._curr_depth = 0

            def __enter__(self):
                self._curr_depth += 1
                return self

            def __exit__(self, exc_type, exc_value, exc_traceback):
                self._curr_depth -= 1
                if not self.is_in_batch():
                    self._state._on_exit_batch()

            def is_in_batch(self):
                return self._curr_depth > 0

        def __init__(self, parent, target_features, memory, rev_memory, tokens, curr_best_score):
            self._parent = parent
            self.target_features = target_features
            self.memory = memory
            self.rev_memory = rev_memory
            self.batch_context = self.BatchContext(self)

            self.print_improvements = True
            tokens = sdutils.change_rev(tokens, True)[0]
            self.best_tokens = None
            self.curr_best_score = None
            self.tmp_best_tokens = None
            self.tmp_best_score = None
            self.set_tokens(tokens, curr_best_score)
            self.rev_ret_stack = []

            self.passes_finished = 0
            self.rev_ret = True

        def set_tmp_state(self, tmp_best_tokens, tmp_best_score):
            tmp_best_tokens = sdutils.change_rev(tmp_best_tokens, True)[0]
            self.tmp_best_score = -float("inf")
            self.update_tokens_if_better(tmp_best_tokens.clone(), tmp_best_score)

        def remove_tmp_state(self):
            self.tmp_best_tokens = self.best_tokens.clone()
            self.tmp_best_score = self.curr_best_score

        def get_end_idx(self):
            return sdutils.find_end_idx(self.tmp_best_tokens)

        def push_rev_ret(self, new_val=False):
            self.rev_ret_stack.append(self.rev_ret)
            self.rev_ret = new_val

        def get_best(self, pop_rev_ret=False):
            if pop_rev_ret:
                self.rev_ret = self.rev_ret_stack.pop()
            to_ret = self.tmp_best_tokens
            if self.rev_ret:
                to_ret = sdutils.rev_tokens(self.tmp_best_tokens)[0]
            return to_ret, self.tmp_best_score

        def get_score(self):
            return self.tmp_best_score

        def get_best_score(self):
            return self.curr_best_score

        def get_tokens(self):
            return self.tmp_best_tokens

        def get_best_tokens(self):
            return self.best_tokens

        def set_tokens(self, tokens, score):
            self.best_tokens = tokens.long()
            self.curr_best_score = score

            self.tmp_best_tokens = tokens.long()
            self.tmp_best_score = score

        def update_tokens_if_better(self, new_tokens, new_best):
            if isinstance(new_tokens, list):
                new_tokens = new_tokens[0].view(-1)
            if isinstance(new_best, list):
                new_best = new_best[0]

            is_improved = _is_improvement_eps(new_best, self.tmp_best_score)
            if is_improved:
                self.tmp_best_score = new_best
                self.tmp_best_tokens = new_tokens.long().clone()
                prefix = "Temp best"
                if _is_improvement_eps(new_best, self.curr_best_score):
                    self.curr_best_score = self.tmp_best_score
                    self.best_tokens = self.tmp_best_tokens.clone()
                    prefix = "Current best"

                if not self.batch_context.is_in_batch():
                    self._parent._print_result(self.target_features, new_tokens, new_best, result_prefix=prefix)

            return is_improved

        def batch(self):
            return self.batch_context

        def _on_exit_batch(self):
            if self.tmp_best_score != self.curr_best_score:
                prefix = "Temp best"
            else:
                prefix = "Current best"

            self._parent._print_result(
                self.target_features, self.tmp_best_tokens, self.tmp_best_score, result_prefix=prefix
            )

    def __init__(
        self,
        tokens_model: torch.nn.Module,
        rating_model: torch.nn.Module,
        clip_model: ClipModel,
        config: PromptUpgraderConfig = None,
    ):
        self._tokens_model = tokens_model
        self._clip_model = clip_model
        self._rating_model = rating_model
        if config is None:
            config = PromptUpgraderConfig()
        self._config = config

        calculator = CombinedClipRatingCalculator(
            self._clip_model, self._rating_model, rating_weight=config.rating_weight
        )
        self._calculator = calculator

    def create_state(self, target_features, prompt, memory=None, rev_memory=None):
        self._verbose_print("Creating new state")
        target_features = target_features.view(1, -1).float()
        target_features /= target_features.norm(dim=-1, keepdim=True)

        if isinstance(prompt, str):
            tokens = clip.tokenize(prompt, truncate=True)[0].cuda()
        else:
            tokens = prompt.clone()

        tokens = tokens.long()
        if memory is None or rev_memory is None:
            memory, rev_memory = self._tokens_model.features_to_memory(target_features, include_rev=True)
        curr_best_score = self._calculator.score_tokens(target_features, tokens)[0].item()

        return PromptUpgrader.PromptState(self, target_features, memory, rev_memory, tokens, curr_best_score)

    def upgrade(
        self, candidate_cnt=1024, max_tokens=sdconsts.prompt_token_len, target_features=None, prompt=None, state=None
    ):
        with torch.no_grad():
            if not state:
                state = self.create_state(target_features, prompt)

            state.push_rev_ret()
            if self._has_hit_target(state):
                self._verbose_print("Input prompt's sim and rating already hits the targets, nothing to do")
                return state.get_best(True)

            tokens = state.best_tokens
            self._print_result(state.target_features, tokens, state.curr_best_score, result_prefix="Initial prompt")

            min_cands = int(candidate_cnt / self._config.cand_mul)
            max_cands = min(int(candidate_cnt * self._config.cand_mul), self._config.max_cands)
            max_tokens = min(sdconsts.prompt_token_len, max_tokens)

            self._init_upgrade_pass(state, min_cands, max_cands, max_tokens)
            if not self._has_hit_target(state):
                self._core_upgrade_pass(state, min_cands, max_cands, max_tokens)
            if not self._has_hit_target(state):
                self._final_upgrade_pass(state, min_cands, max_cands, max_tokens)

            return state.get_best(True)

    def _init_upgrade_pass(self, state, min_cands, max_cands, max_tokens):
        # Trim the prompt, do several past passes and optionally upgrade the ends with a high counts.
        # Tokens get added either after the triming or at the end based on "add_first". Only add
        # half of the required tokens to allow room for inserting more after other upgrades.
        self._verbose_print("Running initial upgrade pass and insertions")
        self.remove_tokens(state=state)
        end_idx = state.get_end_idx()
        if self._config.add_first:
            self.add_tokens(pass_cands=min_cands, max_tokens=(max_tokens + end_idx) / 2 - 2, state=state)

        with state.batch():
            for cands in self._config.quick_pass_cands:
                self.replace_tokens(cands, state=state)

        if self._config.do_large_cap_pass:
            with state.batch():
                self._verbose_print("Upgrading end tokens with a large candidate count")
                self.replace_tokens(max_cands, start_idx=end_idx - 8, end_idx=end_idx, decay_factor=1.0, state=state)
                self.replace_tokens(min_cands, start_idx=end_idx - 16, end_idx=end_idx, decay_factor=1.0, state=state)
                self._verbose_print("Finished end pass")

                self.replace_tokens(max_cands, end_idx=8, decay_factor=1.0, state=state)
                self.replace_tokens(min_cands, end_idx=16, decay_factor=1.0, state=state)
                self._verbose_print("Finished start pass")

        if not self._config.add_first:
            self.add_tokens(pass_cands=min_cands, max_tokens=(max_tokens + end_idx) / 2 - 2, state=state)

    def _core_upgrade_pass(self, state, min_cands, max_cands, max_tokens):
        # Iterative replace tokens following by adding, trimming and readding tokens.
        self._verbose_print("Running core replacement passes")

        for _ in range(self._config.max_iters):
            old_best = state.get_score()
            self.replace_tokens(min_cands, state=state)
            if not _is_improvement_eps(state.get_score(), old_best):
                break

        self.add_tokens(pass_cands=min_cands, max_tokens=max_tokens, state=state)
        state.remove_tmp_state()
        self.replace_tokens(max_cands, state=state)
        self.remove_tokens(state=state)
        self.add_tokens(pass_cands=min_cands, max_tokens=max_tokens, state=state)

    def _final_upgrade_pass(self, state, min_cands, max_cands, max_tokens):
        # A large replacement fast followed by another iterative replacement loop.
        # Finally, upgrade the caps and do another trim-replace cycle
        self._verbose_print("Finishing upgrade with a large pass followed by another refinement loop")
        self.replace_tokens(num_candidates=max_cands, state=state)

        for _ in range(self._config.max_iters):
            old_best = state.get_score()
            self.replace_tokens(min_cands, state=state)
            if not _is_improvement_eps(state.get_score(), old_best):
                break

        if self._config.do_large_cap_pass:
            with state.batch():
                self._verbose_print("Upgrading end tokens with a large candidate count")
                end_idx = state.get_end_idx()
                self.replace_tokens(max_cands, start_idx=end_idx - 8, end_idx=end_idx, decay_factor=1.0, state=state)
                self.replace_tokens(min_cands, start_idx=end_idx - 16, end_idx=end_idx, decay_factor=1.0, state=state)
                self._verbose_print("Finished end pass")

        self._verbose_print("Final insertion and replacement passes")
        self.remove_tokens(state=state)
        self.add_tokens(pass_cands=min_cands, max_tokens=max_tokens, state=state)
        self.replace_tokens(num_candidates=max_cands, state=state)

    def replace_tokens(
        self,
        num_candidates,
        decay_factor=0.9,
        start_idx=1,
        end_idx=-1,
        target_features=None,
        prompt=None,
        state=None,
    ):
        num_candidates = min(num_candidates, sdconsts.num_tokens - 2)
        start_idx = max(1, start_idx)

        with state.batch():
            with torch.no_grad():
                with torch.autocast(device_type="cuda"):
                    if not state:
                        state = self.create_state(target_features, prompt)

                    state.push_rev_ret()

                    for _ in range(self._config.replacement_iters):
                        self._swap(state=state, start_idx=start_idx, end_idx=end_idx)
                        self._run_upgrade_cycle(
                            num_candidates, decay_factor=decay_factor, state=state, start_idx=start_idx, end_idx=end_idx
                        )

                    self._swap(state=state, start_idx=start_idx, end_idx=end_idx)

                    return state.get_best(True)

    # TODO: Trim based on score instead of sim
    def remove_tokens(self, target_features=None, prompt=None, state=None):
        with torch.no_grad():
            if not state:
                state = self.create_state(target_features, prompt)

            state.push_rev_ret(False)

            start_end_idx = state.get_end_idx()

            trimmed_prompt = sdutils.trim_prompt(
                state.get_tokens(),
                self._clip_model,
                max_to_remove=self._config.max_token_removal,
                thresh=self._config.removal_sim_thresh,
                orig_features=state.target_features,
            )
            trimmed_tokens = clip.tokenize(trimmed_prompt)[0].cuda()
            trimmed_score = self._calculator.score_tokens(state.target_features, trimmed_tokens)[0].item()

            state.set_tokens(trimmed_tokens, trimmed_score)

            num_removed = start_end_idx - state.get_end_idx()
            if num_removed > 0:
                self._verbose_print(f"Trimmed {num_removed} low impact tokens {trimmed_prompt}")

            return state.get_best(True)

    def add_tokens(
        self,
        pass_cands=768,
        quick_pass_cands=64,
        max_tokens=sdconsts.prompt_token_len,
        target_features=None,
        prompt=None,
        state=None,
    ):
        con_failed = 0
        max_con_failed = 5

        with torch.no_grad():
            if not state:
                state = self.create_state(target_features, prompt)

            state.push_rev_ret()

            rem_tokens = int(max_tokens - (state.get_end_idx() - 1))
            if rem_tokens > 0:
                self._verbose_print(f"Adding {rem_tokens} tokens")

            for num_added in range(int(rem_tokens)):
                with state.batch():
                    num_checks = min(state.get_end_idx(), self._config.num_insert_checks)
                    """
                    insert_cand_alpha = min((state.get_end_idx())/self._config.num_insert_checks, 1)
                    insert_cands = insert_cand_alpha*min_insert_cands + (1-insert_cand_alpha)*max_insert_cands
                    """
                    curr_cands = int(self._config.insert_cands / num_checks)
                    old_best_score = state.get_best_score()
                    self._insert_token(state, num_cands=int(curr_cands))
                    self.replace_tokens(quick_pass_cands, state=state)

                    self._verbose_print(f"Added token, {rem_tokens-num_added} remain")

                    if not _is_improvement_eps(state.get_best_score(), old_best_score):
                        self.replace_tokens(pass_cands, state=state)

                    if _is_improvement_eps(state.get_best_score(), old_best_score):
                        con_failed = 0
                    else:
                        con_failed += 1

                    if con_failed >= max_con_failed or self._has_hit_target(state):
                        break

            state.remove_tmp_state()
            return state.get_best(True)

    def _insert_token(self, state, num_cands=32 * 2):
        orig_tokens = state.get_tokens()

        if state.tmp_best_tokens[-1] == sdconsts.sot_token:
            print("Attempted to insert into a full token tensor")
            return state.get_best()

        prompt_end = state.get_end_idx()
        to_test = []
        to_upgrade = list(range(1, prompt_end + 1))

        random.shuffle(to_upgrade)

        for curr_end in to_upgrade[: self._config.num_insert_checks]:
            tokens = orig_tokens.clone()
            if tokens[curr_end] == sdconsts.eot_token:
                continue

            tokens[curr_end + 1 :] = tokens[curr_end:-1].clone()
            tokens[curr_end] = 0

            curr_end_idx = sdutils.find_end_idx(tokens)
            rev_tokens = sdutils.change_rev(tokens, False)[0]

            rev_idx = max(curr_end_idx - curr_end, 1)

            replacement_probs = self._tokens_model.get_next_probs(
                memory=state.memory,
                rev_memory=state.rev_memory,
                tokens=rev_tokens.unsqueeze(0),
                rev_tokens=tokens.unsqueeze(0),
                idx = curr_end,
                ascii_only=self._config.ascii_only,
                no_banned=self._config.no_banned,
            )[0]
            _, candidate_tokens = replacement_probs.topk(num_cands, dim=-1)

            candidate_tokens = candidate_tokens.long()
            for replacement_token in candidate_tokens:
                if replacement_token == sdconsts.sot_token:
                    continue

                new_tensor = tokens.clone()
                new_tensor[curr_end] = replacement_token
                to_test.append(new_tensor)

        if not to_test:
            return state.get_best()
        test_stack = torch.stack(tuple(to_test))
        top_tokens, top_sim = self._calculator.rank(state.target_features, test_stack, 1)
        old_score = state.get_score()
        state.set_tmp_state(top_tokens[0][0], top_sim[0].item())

        return _is_improvement_eps(state.get_score(), old_score)

    def _swap(self, state, start_idx=1, end_idx=-1, max_swaps=10):
        tokens_end = sdutils.find_end_idx(state.get_tokens())
        if end_idx == -1:
            end_idx = tokens_end
        end_idx = min(end_idx, tokens_end)

        swap_idxs = [(src, dst) for src in range(start_idx, end_idx) for dst in range(src, end_idx)]
        improved = False
        for _ in range(max_swaps):
            tokens = state.get_tokens()
            to_test = [tokens.clone()]
            for src, dst in swap_idxs:
                if src == dst or tokens[src] == sdconsts.sot_token or tokens[dst] == sdconsts.sot_token:
                    continue
                new_tensor = tokens.clone()
                new_tensor[src], new_tensor[dst] = new_tensor[dst], new_tensor[src]
                to_test.append(new_tensor)

            # TODO: Fix duplcate work
            test_stack = torch.stack(tuple(to_test))
            top_tokens, top_sim = self._calculator.rank(state.target_features, test_stack, 1)

            if state.update_tokens_if_better(top_tokens, top_sim):
                improved = True
            else:
                break

        return improved

    def _run_upgrade_cycle(
        self, num_cands, state, ave_tokens=15, decay_factor=1.0, print_freq=0, start_idx=1, end_idx=-1
    ):
        start_idx = max(1, start_idx)

        tokens, curr_best_score = state.get_best()
        last_idx = sdutils.find_end_idx(tokens)
        if end_idx == -1:
            end_idx = last_idx

        if end_idx > last_idx:
            to_upgrade = list(range(start_idx, last_idx)) + list(range(1, max(start_idx - last_idx, 2)))
        else:
            to_upgrade = list(range(start_idx, end_idx))
        to_upgrade = [last_idx - idx for idx in to_upgrade]

        orig_to_upgrade = list(to_upgrade)
        impact = sdutils.rank_token_impact(
            tokens, self._clip_model, idxs=orig_to_upgrade, target_features=state.target_features
        )
        to_upgrade = [(impact_idx, arg[1]) for impact_idx, arg in enumerate(impact[::-1])]

        ends = []
        total_iters = 0
        iter_num = 0
        num_checks = 0

        # at 10 = 35 * (1 - 35/10) = -87.5
        # at 35 = 35 * (1 - 35/35) = 0
        # at 70 = 35 * (1 - 35/70) = 17.5
        # at 1000 = 35 * (1 - 35/1000) = 33.775
        weight_mul = ave_tokens * (1 - (ave_tokens / max(last_idx, 1)))

        while iter_num < len(to_upgrade):
            impact_idx, curr_end = to_upgrade[iter_num]
            total_iters += 1
            iter_num += 1

            # Don't skip ave_tokens on average prefering low impact tokens
            choice_chance = impact[impact_idx].impact * weight_mul
            skip = choice_chance < random.random()
            if skip or tokens[curr_end] in {sdconsts.sot_token, sdconsts.eot_token}:
                continue
            num_checks += 1
            if self._config.verbose and print_freq != 0 and (curr_end - 1) % print_freq == 0:
                print(f"Finished {curr_end} tokens with {num_cands} candidates per token")
                self._print_result(state.target_features, tokens, curr_best_score, True)

            rev_tokens = sdutils.change_rev(tokens, False)[0]

            replacement_probs = self._tokens_model.get_next_probs(
                memory=state.memory,
                rev_memory=state.rev_memory,
                tokens=rev_tokens.unsqueeze(0),
                rev_tokens=tokens.unsqueeze(0),
                idx = curr_end,
                ascii_only=self._config.ascii_only,
                no_banned=self._config.no_banned,
            )[0]
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
            top_tokens, top_sim = self._calculator.rank(state.target_features, test_stack, 1)

            if state.update_tokens_if_better(top_tokens[0][0], top_sim[0]):
                ends.append(curr_end)
                tokens, curr_best_score = state.get_best()

                # print(iter_num, len(to_upgrade))

                impact = sdutils.rank_token_impact(
                    tokens, self._clip_model, idxs=orig_to_upgrade, target_features=state.target_features
                )

                rem = len(to_upgrade) - iter_num
                to_upgrade = [(impact_idx, arg[1]) for impact_idx, arg in enumerate(impact[:rem:-1])]
                iter_num = 0
                # print(f"New best at token {curr_end} with {num_cands} candidates per token {len(to_upgrade)}")
                # self._print_result(target_features, tokens, curr_best_score, True)

                if decay_factor == 0 or not to_upgrade:
                    break

                num_cands = int(max(128, int(num_cands * decay_factor)))

        return state.update_tokens_if_better(tokens, curr_best_score)

    def set_verbose(self, verbose):
        self._config.verbose = verbose

    def _has_hit_target(self, state):
        top_features = self._clip_model.features_from_tokens(state.get_best_tokens(), verbosity=0)
        top_sim = self._clip_model.cosine_similarity(state.target_features, top_features)[0]
        top_rating = self._rating_model(top_features)[0].item()

        return top_sim >= self._config.target_sim and top_rating >= self._config.target_rating

    def _print_result(self, target_features, tokens, curr_best_score, result_prefix="Current best"):
        if not self._config.verbose:
            return

        tokens = sdutils.change_rev(tokens, False)[0]
        top_features = self._clip_model.features_from_tokens(tokens, verbosity=0)
        top_sim = self._clip_model.cosine_similarity(target_features, top_features)[0]
        top_rating = self._rating_model(top_features)[0].item()
        print(f"{result_prefix} {curr_best_score}. Cosine Similarity {top_sim}. Rating {top_rating}:")
        print(f"{sdutils.decode_tokens(tokens)[0]}\n")

    def _safe_log(self, tensor):
        return torch.log(tensor + 1e-6)

    # TODO: log levels
    def _verbose_print(self, msg):
        if not self._config.verbose:
            return
        print(msg)


def _is_improvement_eps(new_val, prev_val, improve_eps=1e-4):
    if isinstance(new_val, list):
        new_val = new_val[0]
    return new_val - prev_val > improve_eps
