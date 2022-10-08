import ai.stabledisco.constants as sdconsts
import ai.torchmodules as torchmodules
import ai.torchmodules.layers as torchlayers
import ai.torchmodules.scheduler as torchscheduler
import ai.torchmodules.utils as torchutils
import torch
import torch.nn as nn
from ai.stabledisco.decoderpipeline.lowerfeaturelayers import \
    LowerFeatureLayers


class IsTextFeaturesModel(torchmodules.BaseModel):
    name = "IsTextFeatureV3"
    def __init__(self, max_lr=2e-4, min_lr_divisor=20, epoch_batches=32000, step_size_up_epoch_mul=0.5, warmup_period_epoch_mul=2, gamma=0.75, last_epoch=-1, device=None):
        super().__init__(IsTextFeaturesModel.name, device=device)
        # Input = (-1, 77, num_features)
        # Output = (-1, 77, vocab_size)
        # Loss = diffable encoding

        # TODO: Extract to class
        num_res_blocks = 4
        res_unit_mul = 8
        res_layers = 8
        units_div = 2
        dropout_div = 2
        start_dropout = 0.3
        
        self._resblocks = nn.ModuleList()
        curr_width = sdconsts.feature_width
        curr_dropout = start_dropout
        prev_width = curr_width
        for _ in range(num_res_blocks):
            block = torchlayers.ResDenseStack(curr_width, curr_width*res_unit_mul, layers=res_layers, activation=nn.LeakyReLU, dropout=curr_dropout)
            self._resblocks.append(block)
            
            prev_width = curr_width
            curr_width //= units_div
            
            reducer = torchlayers.LinearWithActivation(
                prev_width,
                curr_width,
                dropout=curr_dropout,
                batch_norm_type=None,
            )
            
            self._resblocks.append(reducer)
            
            curr_dropout /= dropout_div
                             
        self._prob_out = torch.nn.Linear(curr_width, 1)
        nn.init.xavier_uniform_(self._prob_out.weight)
        
        self._loss_func = torch.nn.MSELoss()

        self._optimizer = torch.optim.NAdam(
            self.parameters(), max_lr, betas=(0.89, 0.995)
        )

        
        self._scheduler = torchscheduler.make_cyclic_with_warmup(optimizer=self._optimizer,
                                                                 epoch_batches=epoch_batches,
                                                                 max_lr=max_lr,
                                                                 min_lr_divisor=min_lr_divisor,
                                                                 step_size_up_epoch_mul=step_size_up_epoch_mul,
                                                                 warmup_period_epoch_mul=warmup_period_epoch_mul,
                                                                 gamma=gamma,
                                                                 last_epoch=last_epoch,
                                                                 cycle_momentum=False)


    def _calc_batch_loss(self, x_inputs, y_targets):
        with torch.autocast(device_type="cuda"):
            outputs = self(x_inputs)
            return self._loss_func(outputs.view(-1, 1), y_targets.float().view(-1, 1))

    def forward(self, features):
        x = features.float() / features.norm(dim=-1, keepdim=True)
        for layer in self._resblocks:
            x = layer(x)
            
        return self._prob_out(x)
    
    def _get_forward(self, features):
        return self(features).reshape(-1)

    def get_text_prob(self, features, shift_ceil = 1.0):
        percent_ceil = torch.abs(self(features)) / shift_ceil
        predicted_shift = 1 - torch.minimum(percent_ceil, torch.tensor([1.0], device=features.device))
        return predicted_shift.reshape(-1)

    def improve_text_prob(self, features, target_prob=0.96, max_diff=0.03, per_step=0.01,  alpha=0.7, max_divs=10, verbose=False):
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
                dx = -torch.autograd.functional.jacobian(self._get_forward, out_features.float(), create_graph=True).reshape(out_features.shape).float()
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
                
                if cosine_change > max_diff:
                    out_features = prev_out_features
                    cosine_change = 0
                    per_step *= alpha
                    num_divs += 1
           
            if verbose:
                cosine_change = abs(1.0 - (features.unsqueeze(0) @ best_out_features.T))
                print("End Prob", best_out_score)
                print("Cosine Diff of New Features", cosine_change)

            return best_out_features.squeeze(0)
