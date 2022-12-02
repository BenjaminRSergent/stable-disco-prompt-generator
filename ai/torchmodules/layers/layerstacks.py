import torch.nn as nn
from ai.torchmodules.layers.basiclayers import LinearWithActivation, Normalization, QuickGELU, ResLinear
from ai.torchmodules.basemodel import BaseModel


class ReducingResDenseStack(BaseModel):
    def __init__(
        self,
        input_size,
        num_res_blocks=4,
        res_unit_mul=8,
        res_layers=8,
        units_div=2,
        dropout_div=2,
        start_dropout=0.1,
        batch_norm_type=Normalization.NormType.BATCH,
        activation=QuickGELU,
    ) -> None:
        super().__init__()

        self._resblocks = nn.Sequential()
        curr_width = input_size
        curr_dropout = start_dropout
        prev_width = curr_width
        for _ in range(num_res_blocks):
            block = ResDenseStack(
                curr_width,
                int(curr_width * res_unit_mul),
                layers=res_layers,
                activation=activation,
                dropout=curr_dropout,
                batch_norm_type=batch_norm_type,
            )
            self._resblocks.append(block)

            prev_width = curr_width
            curr_width = int(curr_width / units_div)
            # For systems that perform better on multiples of 8
            curr_width -= curr_width % 8

            reducer = LinearWithActivation(
                int(prev_width), int(curr_width), dropout=curr_dropout, batch_norm_type=None, activation=activation
            )

            self._resblocks.append(reducer)

            curr_dropout /= dropout_div

        self.in_features = input_size
        self.out_features = curr_width

    def forward(self, x_inputs):
        return self._resblocks(x_inputs)


class ResDenseStack(BaseModel):
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

        self.in_features = input_size
        self.out_features = input_size

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

class MixedResDenseStack(BaseModel):
    def __init__(
        self,
        input_size,
        units_list,
        batch_norm_type=Normalization.NormType.BATCH,
        activation=nn.LeakyReLU,
        dropout=0.2,
    ) -> None:
        super().__init__()
        self._layers = nn.ModuleList()

        if type(units_list) is int:
            units_list = [(1, units_list)]

        num_layers = len(units_list)

        self.in_features = input_size
        self.out_features = units_list[-1][1]

        for layer_details in units_list:
            if not isinstance(layer_details, tuple):
                layer_details = (1, layer_details)
            cnt, units = layer_details

            for _ in range(cnt):
                self._layers.append(
                    ResLinear(
                        input_size,
                        units,
                        batch_norm_type,
                        activation,
                        dropout=dropout,
                    )
                )

        proj_std = (input_size**-0.5) * ((2 * num_layers) ** -0.5)
        fc_std = (2 * input_size) ** -0.5
        for block in self._layers:
            nn.init.normal_(block._layers.first_layer.weight, std=fc_std)
            nn.init.normal_(block._layers.projection.weight, std=proj_std)

    def forward(self, x_inputs):
        curr_data = x_inputs
        for layer in self._layers:
            curr_data = layer(curr_data)

        return curr_data


class DenseStack(BaseModel):
    def __init__(
        self,
        input_size,
        units_list,
        batch_norm_type=Normalization.NormType.BATCH,
        activation=nn.LeakyReLU,
        dropout=0.2,
    ) -> None:
        super().__init__()
        self._layers = nn.ModuleList()

        if type(units_list) is int:
            units_list = [(1, units_list)]

        self.in_features = input_size
        self.out_features = units_list[-1][1]

        last_units = input_size
        for layer_details in units_list:
            if not isinstance(layer_details, tuple):
                layer_details = (1, layer_details)
            cnt, units = layer_details

            for _ in range(cnt):
                self._layers.append(
                    LinearWithActivation(
                        last_units,
                        units,
                        batch_norm_type,
                        activation,
                        dropout=dropout,
                    )
                )
                last_units = units

    def forward(self, x_inputs):
        curr_data = x_inputs
        for layer in self._layers:
            curr_data = layer(curr_data)

        return curr_data
