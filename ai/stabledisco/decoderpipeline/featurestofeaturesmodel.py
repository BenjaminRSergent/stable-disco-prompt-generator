import ai.torchmodules as torchmodules
import ai.torchmodules.layers as torchlayers
import ai.torchmodules.utils as torchutils
import torch
import torch.nn as nn


class FeaturesToFeaturesModel(torchmodules.BaseModel):
    def __init__(self, device=None, transformer_width=768):
        super().__init__("FeaturesToFeaturesV1")
        # Input = (-1, num_features)
        # Output = (-1, num_features)
        # Loss = diffable encoding

        self._dtype = torch.float
        if device is None:
            device = torchutils.get_default_device()
        self._device = device
        self._transformer_width = transformer_width

        self._dense_stacks = nn.ModuleList()
        dense_stack_units = [[(2, 8096)], [(3, 4048)], [(3, 2024)], [(3, 1024)]]

        last_units = self._transformer_width
        for stack_units in dense_stack_units:
            self._dense_stacks.append(
                torchlayers.ResDenseStack(
                    stack_units,
                    last_units,
                    batch_norm_type=torchlayers.Normalization.NormType.BATCH,
                )
            )
            last_units = stack_units[-1][1]

        self._features_out = torch.nn.Linear(last_units, self._transformer_width)
        nn.init.xavier_uniform_(self._features_out.weight)

        self._loss_func = torchmodules.cosine_loss
        base_learning = 0.002
        self._optimizer = torch.optim.AdamW(
            self.parameters(), base_learning, (0.85, 0.97)
        )
        self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self._optimizer, T_max=4, eta_min=base_learning / 10
        )

    def _calc_batch_loss(self, x_inputs, y_targets):
        outputs = self(x_inputs)
        return self._loss_func(outputs, y_targets)

    def forward(self, x_inputs):
        x_inputs = x_inputs / x_inputs.norm(dim=-1, keepdim=True)
        curr_data = x_inputs
        for layer in self._dense_stacks:
            curr_data = layer(curr_data)

        return self._features_out(curr_data)
