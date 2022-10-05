import ai.stabledisco.constants as sdconsts
import ai.torchmodules as torchmodules
import ai.torchmodules.layers as torchlayers
import torch
import torch.nn as nn


class FeaturesToRatingModel(torchmodules.BaseModel):
    def __init__(self, base_learning=2e-4, learning_divisor=10, step_size_up=25500, device=None):
        super().__init__("FeaturesToRatingV5", device=device)
        # Input = (-1, 77, num_features)
        # Output = (-1, 77, vocab_size)
        # Loss = diffable encoding

        self._res_stack = torchlayers.ResDenseStack(sdconsts.feature_width, sdconsts.feature_width*6, layers=4, activation=nn.LeakyReLU, dropout=0.1)

        dense_stack_units = [(1, 8192),
                             (2, 4096),
                             (5, 1024),
                             (8, 512)]
                             
        self._dense_stack = torchlayers.DenseStack(
            sdconsts.feature_width,
            dense_stack_units,
            activation=nn.LeakyReLU,
            dropout=0.0,
        )
        
        self._rating_lin = torch.nn.Linear(self._dense_stack.out_features, 1)
        nn.init.xavier_uniform_(self._rating_lin.weight)

        self._loss_func = nn.MSELoss()

        self._optimizer = torch.optim.NAdam(
            self.parameters(), base_learning, betas=(0.88, 0.995)
        )

        self._scheduler = torch.optim.lr_scheduler.CyclicLR(
            self._optimizer,
            base_lr=base_learning / learning_divisor,
            max_lr=base_learning,
            step_size_up=step_size_up,
            mode="triangular2",
            cycle_momentum=False,
        )

    def _calc_batch_loss(self, x_inputs, y_targets):
        with torch.autocast(device_type="cuda"):
            outputs = self(x_inputs)
            return self._loss_func(outputs.view(-1, 1), y_targets.view(-1, 1))

    def forward(self, features):
        features = features / features.norm(dim=-1, keepdim=True)
        x = self._res_stack(features)
        x = self._dense_stack(x)
        return self._rating_lin(x)

    def get_rating(self, features):
        return self(features).reshape(1)

    def rank_ratings(self, 
                     tokens,
                     top_count=None,
                     largest=True
    ):
        if top_count is None:
            top_count = len(tokens)

        top_count = min(top_count, len(tokens))
        with torch.no_grad():
            
            features = self._clip_model.encode_text(tokens)
            ratings = self(features).view(-1)
        

            top_ratings, top_labels = (
                ratings.float().cpu().topk(top_count, dim=-1, largest=largest)
            )
            top_words = [
                tokens[top_labels[idx].numpy()] for idx in range(top_count)
            ]

            return top_words, top_ratings

    def improve_rating(self, features, target_rating=9.0, max_diff=0.05, per_step=0.01,  alpha=0.5, patience=5, max_divs=10, verbose=False):
        with torch.no_grad():
            if len(features.shape) == 1:
                features = features.view(1, -1)
            features = features / features.norm(dim=-1, keepdim=True)
            before_rating = self.get_rating(features)
            out_features = torch.clone(features)
            cosine_change = 0
            if verbose:
                print("Start rating", before_rating)

            best_out_features = out_features.clone()
            best_out_score = before_rating

            prev_rating = before_rating
            con_worse = 0
            num_divs = 0
            while self.get_rating(out_features)[0] <= target_rating and cosine_change < max_diff and num_divs < max_divs:
                dx = torch.autograd.functional.jacobian(self.get_rating, out_features.float(), create_graph=True).reshape(out_features.shape)

                dx_norm = dx / dx.norm(dim=-1, keepdim=True)

                out_features += per_step * dx_norm
                out_features = out_features / out_features.norm(dim=-1, keepdim=True)

                mid_rating = self.get_rating(out_features)
                if mid_rating < prev_rating:
                    con_worse += 1
                

                if con_worse > patience:
                    per_step *= alpha
                    num_divs += 1
                    con_worse = 0
                prev_rating = mid_rating

                
                cosine_change = abs(1.0 - (features.unsqueeze(0) @ out_features.T))

                if mid_rating > best_out_score and cosine_change < max_diff:
                    best_out_score = mid_rating
                    best_out_features = out_features.clone()

            if verbose:
                cosine_change = abs(1.0 - (features.unsqueeze(0) @ best_out_features.T))
                print("End Rating", best_out_score)
                print("Cosine Diff of New Features", cosine_change)

            return best_out_features.squeeze(0)
