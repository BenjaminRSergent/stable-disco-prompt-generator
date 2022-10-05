import ai.stabledisco.constants as sdconsts
import ai.torchmodules as torchmodules
import ai.torchmodules.layers as torchlayers
import ai.torchmodules.utils as torchutils
import torch
import torch.nn as nn
from ai.stabledisco.decoderpipeline.lowerfeaturelayers import \
    LowerFeatureLayers


class IsTextFeaturesModel(torchmodules.BaseModel):
    def __init__(self, base_learning = 1e-4, learning_divisor=6, step_size_up=20000, device=None):
        super().__init__("IsTextFeatureV2", device=device)
        # Input = (-1, 77, num_features)
        # Output = (-1, 77, vocab_size)
        # Loss = diffable encoding

        self._transformer_width = sdconsts.feature_width
        self._res_stack = torchlayers.ResDenseStack(sdconsts.feature_width, sdconsts.feature_width*6, layers=4, activation=nn.LeakyReLU, dropout=0.1)
        
        dense_stack_units = [(4, 2048),
                        (4, 4096),
                        (1, sdconsts.feature_width)]
                             
        self._expanding_dense_stack = torchlayers.DenseStack(
            sdconsts.feature_width,
            dense_stack_units,
            activation=nn.LeakyReLU,
            dropout=0.00,
        )
        

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
        
        self._prob_out = torch.nn.Linear(self._dense_stack.out_features, 1)
        nn.init.xavier_uniform_(self._prob_out.weight)
        
        self._sig = torch.nn.Sigmoid()
        self._loss_func = torch.nn.BCEWithLogitsLoss()

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
        x = self._res_stack(features.float())
        x = self._expanding_dense_stack(x)
        x = self._dense_stack(x)
        return self._prob_out(x)

    def get_text_prob(self, features):
        return self._sig(self(features)).reshape(-1)

    def improve_text_prob(self, features, target_prob=0.96, max_diff=0.03, per_step=0.05,  alpha=0.7, max_divs=40, verbose=False):
        # TODO: Extract common code with improve rating
        with torch.no_grad():
            if len(features.shape) == 1:
                features = features.view(1, -1)
            features = features / features.norm(dim=-1, keepdim=True)
            before_prob = self.get_text_prob(features)
            out_features = torch.clone(features)
            cosine_change = 0
            if verbose:
                print("Start prob", before_prob)

            best_out_features = out_features.clone()
            best_out_score = before_prob

            prev_prob = before_prob
            con_worse = 0
            num_divs = 0
            
            eps_scalar = torch.tensor([1e-33], device=features.device)
            while self.get_text_prob(out_features)[0] <= target_prob and cosine_change < max_diff and num_divs < max_divs:
                dx = torch.autograd.functional.jacobian(self.get_text_prob, out_features.float(), create_graph=True).reshape(out_features.shape).float()

                
                dx_mag = dx.norm(dim=-1, keepdim=True)
                if dx_mag < eps_scalar:
                    dx /= eps_scalar
                else:
                    dx /= dx_mag
                
                dx_shift = per_step * dx
                
                prev_out_features = out_features.clone()
                out_features += dx_shift
                out_features = out_features / out_features.norm(dim=-1, keepdim=True)

                mid_prob = self.get_text_prob(out_features)
                if mid_prob < prev_prob:
                    con_worse += 1
                

                prev_prob = mid_prob
                cosine_change = abs(1.0 - (features.unsqueeze(0) @ out_features.T))

                if mid_prob > best_out_score and cosine_change < max_diff:
                    if mid_prob - best_out_score < 1e-5:
                        per_step /= alpha
                    best_out_score = mid_prob
                    best_out_features = out_features.clone()
                elif cosine_change > max_diff:
                    out_features = prev_out_features
                    cosine_change = 0
                    per_step *= alpha
                    num_divs += 1
           
            if verbose:
                cosine_change = abs(1.0 - (features.unsqueeze(0) @ best_out_features.T))
                print("End Prob", best_out_score)
                print("Cosine Diff of New Features", cosine_change)

            return best_out_features.squeeze(0)
