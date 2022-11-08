import abc
from functools import lru_cache

import ai.stabledisco.utils as sdutils
import torch
from ai.stabledisco.clipmodel import ClipModel


class MetricsCalculator(abc.ABC):
    def rank(self, target_features, tokens, top_count):
        scores = self.score_tokens(target_features, tokens)
        top_scores, top_labels = scores.float().cpu().topk(top_count, dim=-1)
        top_tokens = [tokens[[top_labels][i].numpy()] for i in range(top_count)]
        top_scores = [top_scores[i].numpy() for i in range(top_count)]

        return top_tokens, top_scores

    def arg_sort(self, target_features, tokens, top_count):
        scores = self.score_tokens(target_features, tokens)

        top_labels = torch.argsort(scores, dim=-1)
        top_scores = [scores[i] for i in range(top_count)]

        return top_labels, top_scores

    @abc.abstractmethod
    def score_tokens(self, target_features, tokens):
        pass


class ClipCalculator(MetricsCalculator):
    def __init__(self, clip_model: ClipModel) -> None:
        super().__init__()
        self._clip_model = clip_model

    def score_tokens(self, target_features, tokens):
        token_features = self._clip_model.features_from_tokens(tokens)
        return self._clip_model.cosine_similarity(target_features, token_features).unsqueeze(0)


class RatingCalculator(MetricsCalculator):
    def __init__(
        self,
        to_rating_model: torch.nn.Module,
    ) -> None:
        super().__init__()
        self._to_rating_model = to_rating_model

    def score_tokens(self, _, tokens):
        token_features = self._to_rating_model.features_from_tokens(tokens)
        return self._to_rating_model(token_features).view(-1)


class CombinedClipRatingCalculator(MetricsCalculator):
    def __init__(
        self,
        clip_model: ClipModel,
        to_rating_model: torch.nn.Module,
        rating_weight=1.0,
        clip_weight=1.0,
        min_target_rating=7.75,
    ) -> None:
        super().__init__()
        self._clip_model = clip_model
        self._to_rating_model = to_rating_model
        self._rating_weight = rating_weight
        self._clip_weight = clip_weight
        self._target_rating = None
        self._min_target_rating = min_target_rating

    def score_tokens(self, target_features, tokens, target_rating=None):
        tokens = sdutils.change_rev(tokens, False).view(tokens.shape)
        token_features = self._clip_model.features_from_tokens(tokens, verbosity=0)
        # Reward a 0.055 increase in similarity the same as a 1.0 increase in rating at baseline.
        # That scale roughly maps to typical values and changes during evolution
        sim_step_scale = 0.02 / self._clip_weight

        # Only look at top percent
        # The typical starting point for a decent prompt is cosine sim 0.45
        sim_floor = 0.5
        sim_fitness = self._clip_model.cosine_similarity(target_features, token_features).unsqueeze(0)
        sim_fitness = self._scale_fitness_linear(sim_fitness, mid_val=sim_floor, step=sim_step_scale, max_val=1.0)

        # The typical starting point for a decent rating is 7.5
        if target_rating is None:
            target_rating = max(self._get_target_rating_for(target_features), self._min_target_rating)

        min_rating = 7.0
        mid_rating = target_rating
        max_rating = max(target_rating, 10)
        rating_fitness = self._to_rating_model(token_features).unsqueeze(0)
        rating_fitness = self._rating_weight * self._scale_fitness_decay(
            rating_fitness, min_val=min_rating, mid_val=mid_rating, max_val=max_rating
        )

        return sim_fitness + rating_fitness

    @lru_cache(maxsize=32)
    def _get_target_rating_for(self, target_features):
        return self._to_rating_model(target_features.view(1, -1))[0].item()

    def _scale_fitness_linear(self, fitness, mid_val, step, max_val=1.0):
        # Center on mid-val linearly increasing by step
        ret = (torch.clamp(fitness, max=max_val) - mid_val) / step
        return ret.view(-1)

    def _scale_fitness_decay(self, fitness, min_val, mid_val, max_val):
        # Halve growth after the midval. Fitness is 1 at midval
        fitness = torch.clamp(fitness, max=max_val)
        fitness = sdutils.norm_scalars(fitness, min_val, max_val)

        mid_val = sdutils.norm_scalars(mid_val, min_val, max_val)
        min_val = 0
        max_val = 1
        fitness = sdutils.ease_out_quat(fitness)

        below_idx = fitness < mid_val
        above_idx = fitness >= mid_val

        fitness[below_idx] = 2 * fitness[below_idx] - mid_val
        fitness[above_idx] = fitness[above_idx]

        return fitness.view(-1)
