import ai.torchmodules as torchmodules
import ai.torchmodules.layers as torchlayers
import ai.torchmodules.utils as torchutils
import torch.nn as nn


class LowerFeatureLayers(torchmodules.BaseModel):
    pruned_out_size = 5202

    def __init__(self, transformer_width=768, dropout=0.05, device=None):
        super().__init__("FeatureLayers")

        if device is None:
            device = torchutils.get_default_device()
        self._device = device

        self._transformer_width = transformer_width
        dense_stack_units = [
            (3, self._transformer_width * 2),
            (3, self._transformer_width * 4),
            (1, self._transformer_width * 6),
            (1, self._transformer_width * 8),
        ]

        self.in_features = dense_stack_units[0][1]
        self.out_features = dense_stack_units[-1][1]

        self._dense_stack = torchlayers.DenseStack(
            self._transformer_width,
            dense_stack_units,
            activation=nn.LeakyReLU,
            dropout=dropout,
        )

    def forward(self, x):
        return self._dense_stack(x)
