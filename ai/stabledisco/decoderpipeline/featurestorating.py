import ai.torchmodules as torchmodules
import ai.torchmodules.layers as torchlayers
import ai.torchmodules.utils as torchutils
import torch
import torch.nn as nn


class FeaturesToRatingModel(torchmodules.BaseModel):
    def __init__(self, clip_model, device=None):
        super().__init__("FeaturesToRatingV1")
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

        dense_stack_units = [(1, 8096), (3, 4096), (3, 2048), (3, 1024), (3, 512)]
        self._dense_stack = torchlayers.DenseStack(
            self._transformer_width,
            dense_stack_units,
            activation=nn.LeakyReLU,
            dropout=0,
        )
        self._rating_out = torch.nn.Linear(dense_stack_units[-1][-1], 1)
        nn.init.xavier_uniform_(self._rating_out.weight)

        self._loss_func = nn.MSELoss()

        base_learning = 9e-5
        self._optimizer = torch.optim.NAdam(
            self.parameters(), base_learning, betas=(0.85, 0.985)
        )

        self._scheduler = torch.optim.lr_scheduler.CyclicLR(
            self._optimizer,
            base_lr=base_learning / 20,
            max_lr=base_learning,
            step_size_up=5000,
            mode="triangular",
            cycle_momentum=False,
        )

    def _calc_batch_loss(self, x_inputs, y_targets):
        with torch.autocast(device_type="cuda"):
            outputs = self(x_inputs)
            return self._loss_func(outputs.view(-1, 1), y_targets.view(-1, 1))

    def forward(self, features):
        features = features / features.norm(dim=-1, keepdim=True)
        x = self._dense_stack(features.float())

        return self._rating_out(x)

    def get_rating(self, features):
        return self(features).reshape(1)

    def improve_rating(self, features, target_rating=8.75, max_diff=0.02, per_step=0.004, verbose=False):
        with torch.no_grad():
            if len(features.shape) == 1:
                features = features.view(1, -1)
            features = features / features.norm(dim=-1, keepdim=True)
            before_rating = self.get_rating(features)
            out_features = torch.clone(features)
            cosine_change = 0
            if verbose:
                print("Start rating", before_rating)
            while self.get_rating(out_features)[0] <= target_rating and cosine_change < max_diff:
                dx = torch.autograd.functional.jacobian(self.get_rating, out_features, create_graph=True).reshape(out_features.shape)
                out_features += per_step * dx / dx.norm(dim=-1, keepdim=True)
                out_features = out_features / out_features.norm(dim=-1, keepdim=True)

                cosine_change = abs(1.0 - (features.unsqueeze(0) @ out_features.T))
                

            if verbose:
                print("End Rating", self(out_features))
                print("Cosine Diff of Improve Features", cosine_change)
            return out_features.squeeze(0)
