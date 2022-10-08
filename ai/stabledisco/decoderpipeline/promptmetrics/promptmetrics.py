import abc

import torch


class MetricsCalculator(abc.ABC):
    def rank(self, target_features, tokens, top_count):
        scores = self.score_tokens(target_features, tokens)
        top_scores, top_labels = (
            scores.float().cpu().topk(top_count, dim=-1)
        )
        top_tokens = [
            tokens[[top_labels][i].numpy()] for i in range(top_count)
        ]
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
    def __init__(self, clip_model) -> None:
        super().__init__()
        self._clip_model = clip_model

    def score_tokens(self, target_features, tokens):
        token_features = self._clip_model.features_from_tokens(tokens)
        return self._clip_model.cosine_similarity(target_features, token_features).unsqueeze(0)


class RatingCalculator(MetricsCalculator):
    def __init__(self, clip_model, to_rating_model) -> None:
        super().__init__()
        self._clip_model = clip_model
        self._to_rating_model = to_rating_model

    def score_tokens(self, _, tokens):
        token_features = self._to_rating_model.features_from_tokens(tokens)
        return self._to_rating_model(token_features).view(-1)


class CombinedClipRatingCalculator(MetricsCalculator):
    def __init__(self, clip_model, to_rating_model, rating_weight=1.0, clip_weight=1.0) -> None:
        super().__init__()
        self._clip_model = clip_model
        self._to_rating_model = to_rating_model
        self._rating_weight = rating_weight

        
        self._clip_weight = clip_weight

    def score_tokens(self, target_features, tokens):
        token_features = self._clip_model.features_from_tokens(tokens, verbosity=0)
    
        rating_step_scale = self._rating_weight
        # Reward a 0.035 increase in similarity the same as a 1.0 increase in rating at baseline.
        # That scale roughly maps to typical values and changes during evolution
        sim_step_scale = 0.035 * self._clip_weight
        
        # The typical starting point for a decent prompt is cosine sim 0.45
        sim_floor = 0.5
        sim_fitness = self._clip_model.cosine_similarity(target_features, token_features).unsqueeze(0)
        sim_fitness = self._scale_fitness(sim_fitness, floor=sim_floor, ceil=sim_floor+sim_step_scale, max_val=1.0)
        
        # The typical starting point for a decent rating is 8.0
        rating_floor = 8.15
        rating_fitness = self._to_rating_model(token_features).unsqueeze(0)
        rating_fitness = self._scale_fitness(rating_fitness, floor=rating_floor, ceil=rating_floor+rating_step_scale, max_val=8.0)
        
        return sim_fitness + rating_fitness

    def _scale_fitness(self, fitness, floor, ceil, max_val, power=None):
        span = ceil - floor
        fitness = torch.clamp(fitness, max=max_val)
        ret = (fitness - floor) / span
        if power is not None:
            torch.pow(ret, power)
        
        return ret.view(-1)
