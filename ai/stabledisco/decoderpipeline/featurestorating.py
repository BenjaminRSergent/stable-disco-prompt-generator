import ai.stabledisco.constants as sdconsts
from ai.stabledisco.decoderpipeline.lowerfeaturelayers import LowerFeatureLayers
import ai.stabledisco.utils as sdutils
import ai.torchmodules as torchmodules
import ai.torchmodules.layers as torchlayers
from ai.torchmodules.layers.basiclayers import Normalization
import ai.torchmodules.scheduler as torchscheduler
import torch
import torch.nn as nn
from ai.stabledisco.decoderpipeline.knowledgetransfernetwork import KnowledgeTransferNetwork


class FeaturesToRatingModel(torchmodules.BaseModel):
    name = "FeaturesToRatingV9"

    @staticmethod
    def build_teacher(**kwargs):
        return FeaturesToRatingModel(*kwargs)

    @staticmethod
    def build_student(**kwargs):
        return FeaturesToRatingModel(name_suffix="Student", **kwargs)

    @staticmethod
    def build_knowledge_transfer_model(teacher=None, teacher_checkpoint="best", **kwargs):
        if teacher is None:
            teacher = FeaturesToRatingModel.build_teacher()
            teacher.load_weights(teacher_checkpoint, strict=False)

        student = FeaturesToRatingModel.build_student()
        return KnowledgeTransferNetwork(student, teacher, name_suffix=FeaturesToRatingModel.name, **kwargs)

    # TODO: Decouple learning rate scheduler
    def __init__(
        self,
        name_suffix="",
        max_lr=3e-4,
        min_lr_divisor=6,
        num_res_blocks=4,
        res_unit_mul=4,
        res_layers=4,
        units_div=1.5,
        dropout_div=1.5,
        start_dropout=0.5,
        epoch_batches=8214,
        step_size_up_epoch_mul=2,
        warmup_period_epoch_mul=1,
        gamma=0.5,
        last_epoch=-1,
        device=None,
    ):
        super().__init__(FeaturesToRatingModel.name + name_suffix, device=device)

        self._res_stack = torchlayers.ReducingResDenseStack(
            sdconsts.feature_width,
            num_res_blocks=num_res_blocks,
            res_unit_mul=res_unit_mul,
            res_layers=res_layers,
            units_div=units_div,
            dropout_div=dropout_div,
            start_dropout=start_dropout,
            batch_norm_type=Normalization.NormType.BATCH,
            activation=nn.LeakyReLU,
        )

        self._rating_out = torch.nn.Linear(self._res_stack.out_features, 1)
        nn.init.xavier_uniform_(self._rating_out.weight)

        self._loss_func = nn.MSELoss()
        self._optimizer = torch.optim.AdamW(self.parameters(), max_lr, betas=(0.88, 0.995))

        self._scheduler = torchscheduler.make_cyclic_with_warmup(
            optimizer=self._optimizer,
            epoch_batches=epoch_batches,
            max_lr=max_lr,
            min_lr_divisor=min_lr_divisor,
            step_size_up_epoch_mul=step_size_up_epoch_mul,
            warmup_period_epoch_mul=warmup_period_epoch_mul,
            gamma=gamma,
            last_epoch=last_epoch,
            cycle_momentum=False,
        )

    def _calc_batch_loss(self, x_inputs, y_targets):
        with torch.autocast(device_type="cuda"):
            outputs = self(x_inputs)
            return self._loss_func(outputs.view(-1), y_targets.view(-1))

    def forward(self, features):
        x = features / features.norm(dim=-1, keepdim=True)
        x = self._res_stack(features)
        return self._rating_out(x).view(-1)

    def get_rating(self, features):
        return self(features).reshape(1)

    def rank_ratings(self, tokens, top_count=None, largest=True):
        if top_count is None:
            top_count = len(tokens)

        top_count = min(top_count, len(tokens))
        with torch.no_grad():

            features = self._clip_model.encode_text(tokens)
            ratings = self(features).view(-1)

            top_ratings, top_labels = ratings.float().cpu().topk(top_count, dim=-1, largest=largest)
            top_words = [tokens[top_labels[idx].numpy()] for idx in range(top_count)]

            return top_words, top_ratings

    def improve_rating(
        self, features, target_rating=9.0, max_diff=0.05, per_step=1e-3, alpha=0.85, max_divs=200, verbose=False
    ):
        with torch.no_grad():
            if len(features.shape) == 1:
                features = features.view(1, -1)
            features = features / features.norm(dim=-1, keepdim=True)
            before_rating = self.get_rating(features)
            if before_rating >= target_rating:
                return features
            out_features = torch.clone(features)
            cosine_change = 0
            if verbose:
                print("Start rating", before_rating)

            best_out_features = out_features.clone()
            best_out_score = before_rating

            def get_adjusted_rating(other):
                return self.get_rating(other) + sdutils.cosine_sim(features, other)

            con_better = 0
            num_divs = 0
            while self.get_rating(out_features)[0] < target_rating and cosine_change < max_diff and num_divs < max_divs:
                dx = torch.autograd.functional.jacobian(
                    get_adjusted_rating, out_features.float(), create_graph=True
                ).reshape(out_features.shape)

                dx_norm = dx / dx.norm(dim=-1, keepdim=True)

                prev_out_features = out_features.clone()
                out_features += per_step * dx_norm
                out_features = out_features / out_features.norm(dim=-1, keepdim=True)

                mid_rating = self.get_rating(out_features)

                cosine_change = abs(1.0 - (features.unsqueeze(0) @ out_features.T))

                if mid_rating > best_out_score and cosine_change < max_diff:
                    if mid_rating - best_out_score < 1e-5:
                        per_step /= alpha
                    best_out_score = mid_rating
                    best_out_features = out_features.clone()

                if cosine_change > max_diff:
                    out_features = prev_out_features
                    cosine_change = 0
                    per_step *= alpha
                    num_divs += 1
                    con_better = 0
                else:
                    con_better += 1
                    if con_better > 3:
                        per_step /= alpha

            if verbose:
                cosine_change = abs(1.0 - (features.unsqueeze(0) @ best_out_features.T))
                print("End Rating", best_out_score)
                print("Cosine Diff of New Features", cosine_change)

            return best_out_features.squeeze(0)
