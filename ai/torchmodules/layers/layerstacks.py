import torch.nn as nn
from ai.torchmodules.layers.basiclayers import (
    LinearWithActivation,
    Normalization,
    QuickGELU,
    ResLinear,
)


class ResDenseStack(nn.Module):
    def __init__(
        self,
        input_size,
        block_width,
        layers,
        batch_norm_type=Normalization.NormType.BATCH,
        activation=QuickGELU,
        dropout=0.2,
    ) -> None:
        super().__init__()
        self._layers = nn.ModuleList()

        self.resblocks = nn.Sequential(
            *[
                ResLinear(
                    input_size,
                    block_width,
                    batch_norm_type=batch_norm_type,
                    activation=activation,
                    dropout=dropout,
                )
                for _ in range(layers)
            ]
        )

        proj_std = (input_size**-0.5) * ((2 * layers) ** -0.5)
        fc_std = (2 * input_size) ** -0.5
        for block in self.resblocks:
            nn.init.normal_(block._layers.first_layer.weight, std=fc_std)
            nn.init.normal_(block._layers.projection.weight, std=proj_std)

    def forward(self, x_inputs):
        return self.resblocks(x_inputs)


class DenseStack(nn.Module):
    def __init__(
        self,
        input_size,
        units_list,
        batch_norm_type=Normalization.NormType.BATCH,
        activation=nn.LeakyReLU,
        input_inplace=False,
        dropout=0.2,
    ) -> None:
        super().__init__()
        self._layers = nn.ModuleList()

        if type(units_list) is int:
            units_list = [units_list]

        last_units = input_size
        for layer_details in units_list:
            if not isinstance(layer_details, tuple):
                layer_details = (1, layer_details)
            cnt, units = layer_details

            for _ in range(cnt):
                inplace = len(self._layers) == 0 or input_inplace
                self._layers.append(
                    LinearWithActivation(
                        last_units,
                        units,
                        batch_norm_type,
                        activation,
                        inplace=inplace,
                        dropout=dropout,
                    )
                )
                last_units = units

    def forward(self, x_inputs):
        curr_data = x_inputs
        for layer in self._layers:
            curr_data = layer(curr_data)

        return curr_data
