import ai.torchmodules as torchmodules
import ai.torchmodules.layers as torchlayers
import ai.torchmodules.utils as torchutils
import torch
import torch.nn as nn
from ai.stabledisco.decoderpipeline.lowerfeaturelayers import \
    LowerFeatureLayers


class IsTextFeaturesModel(torchmodules.BaseModel):
    def __init__(self, clip_model, freeze_lower=True, device=None):
        super().__init__("IsTextFeatureV1")
        # Input = (-1, 77, num_features)
        # Output = (-1, 77, vocab_size)
        # Loss = diffable encoding
        print(clip_model.dtype)

        self._dtype = clip_model.dtype
        if device is None:
            device = torchutils.get_default_device()
        self._device = device

        self._clip_model = clip_model
        for param in self._clip_model.parameters():
            param.requires_grad = False

        clip_state_dict = self._clip_model.state_dict()
        self._transformer_width = clip_state_dict["ln_final.weight"].shape[0]
        self._transformer_layers = len(
            set(
                k.split(".")[2]
                for k in clip_state_dict
                if k.startswith("transformer.resblocks")
            )
        )

        self._feature_expander = LowerFeatureLayers(dropout=0.05)

        if freeze_lower:
            self._feature_expander.freeze()
        else:
            self._feature_expander.unfreeze()
        
        dense_stack_units = [(1, 512),
                             (2, 128),
                             (2, 64),
                             (1, 32)]
                             
        self._dense_stack = torchlayers.DenseStack(
            self._feature_expander.out_features,
            dense_stack_units,
            activation=nn.LeakyReLU,
            dropout=0.1,
        )
        
        self._prob_out = torch.nn.Linear(self._dense_stack.out_features, 1)
        nn.init.xavier_uniform_(self._prob_out.weight)

        self._sig = torch.nn.Sigmoid()
        self._loss_func = torch.nn.BCEWithLogitsLoss()

        base_learning = 5e-5
        self._optimizer = torch.optim.NAdam(
            self.parameters(), base_learning, betas=(0.88, 0.995)
        )

        self._scheduler = torch.optim.lr_scheduler.CyclicLR(
            self._optimizer,
            base_lr=base_learning / 6,
            max_lr=base_learning,
            step_size_up=8000,
            mode="triangular",
            cycle_momentum=False,
        )

    def _calc_batch_loss(self, x_inputs, y_targets):
        with torch.autocast(device_type="cuda"):
            outputs = self(x_inputs)
            return self._loss_func(outputs.view(-1, 1), y_targets.view(-1, 1))

    def forward(self, features):
        features = features / features.norm(dim=-1, keepdim=True)
        x = self._feature_expander(features.float())
        x = self._dense_stack(x)
        return self._prob_out(x)

    def get_text_prob(self, features):
        return self._sig(self(features)).reshape(-1)

    def improve_text_prob(self, features, target_prob=0.96, max_diff=0.03, per_step=0.005, alpha=0.95, patience=10, max_divs = 40, verbose=False):
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
            while self.get_text_prob(out_features)[0] <= target_prob and cosine_change < max_diff and num_divs < max_divs:
                dx = torch.autograd.functional.jacobian(self.get_text_prob, out_features.float(), create_graph=True).reshape(out_features.shape)

                dx_norm = dx / dx.norm(dim=-1, keepdim=True)

                out_features += per_step * dx_norm
                out_features = out_features / out_features.norm(dim=-1, keepdim=True)

                mid_prob = self.get_text_prob(out_features)
                if mid_prob < prev_prob:
                    con_worse += 1
                

                if con_worse > patience:
                    per_step *= alpha
                    num_divs += 1
                    con_worse = 0
                prev_prob = mid_prob

                
                cosine_change = abs(1.0 - (features.unsqueeze(0) @ out_features.T))

                if mid_prob > best_out_score and cosine_change < max_diff:
                    best_out_score = mid_prob
                    best_out_features = out_features.clone()

            if verbose:
                cosine_change = abs(1.0 - (features.unsqueeze(0) @ best_out_features.T))
                print("End Prob", best_out_score)
                print("Cosine Diff of New Features", cosine_change)

            return best_out_features.squeeze(0)
