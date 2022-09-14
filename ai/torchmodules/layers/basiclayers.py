from collections import OrderedDict
from enum import Enum

import torch
import torch.nn as nn


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class IdentityLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_input):
        return x_input


class Reshaper(nn.Module):
    def __init__(self, out_shape):
        super().__init__()
        self._out_shape = out_shape

    def forward(self, x_inputs):
        return x_inputs.view(self._out_shape)


class Normalization(nn.Module):
    class NormType(Enum):
        NONE = (0,)
        BATCH = (1,)
        LAYER = 2

    def __init__(self, norm_type: NormType, output_shape=None):
        super().__init__()

        if not norm_type or norm_type.value == self.NormType.NONE.value:
            self._batch_layer = IdentityLayer()
        elif norm_type.value == self.NormType.BATCH.value:
            if output_shape is None:
                raise Exception(
                    "Batch norm requires an output shape, not including batch dimension"
                )
            self._batch_layer = nn.BatchNorm1d(output_shape)
        elif norm_type.value == self.NormType.LAYER.value:
            if output_shape is None:
                raise Exception(
                    "Layer norm requires an output shape, not including batch dimension"
                )
            self._batch_layer = nn.LayerNorm(output_shape)
        else:
            raise Exception(f"Bad val {norm_type}")

    def forward(self, x_inputs):
        return self._batch_layer(x_inputs)


class LinearWithActivation(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        batch_norm_type=Normalization.NormType.BATCH,
        activation="QuickGELU",
        inplace=False,
        dropout=0.20,
    ):
        super().__init__()
        self._norm = Normalization(batch_norm_type, output_size)

        if dropout != 0:
            dropout_layer = nn.Dropout(dropout)
        else:
            dropout_layer = IdentityLayer()

        if activation is None:
            activation = IdentityLayer
        elif activation == "QuickGELU":
            activation = QuickGELU

        self._layers = nn.Sequential(
            OrderedDict(
                [
                    (
                        "first_layer",
                        nn.Linear(
                            input_size,
                            output_size,
                            bias=(batch_norm_type == Normalization.NormType.NONE),
                        ),
                    ),
                    ("activation", activation()),
                    ("norm", Normalization(batch_norm_type, output_size)),
                    ("dropout", dropout_layer),
                ]
            )
        )

        nn.init.xavier_uniform_(self._layers.first_layer.weight)

    def forward(self, x):
        return self._layers(x)

    def get_input_size(self):
        return self._layer.first_layer.in_features

    def get_output_size(self):
        return self._layer.first_layer.out_features


class ResAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, seq_len: int):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = self.build_attention_mask(seq_len)

    def attention(self, x: torch.Tensor):
        self.attn_mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device)
            if self.attn_mask is not None
            else None
        )
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

    def build_attention_mask(self, seq_len):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(seq_len, seq_len)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask


class ResLinear(nn.Module):
    def __init__(
        self,
        input_size,
        block_width,
        batch_norm_type=Normalization.NormType.BATCH,
        activation=QuickGELU,
        dropout=0.20,
    ):
        super().__init__()

        if activation is None:
            activation = IdentityLayer()

        if dropout != 0:
            dropout_layer = nn.Dropout(dropout)
        else:
            dropout_layer = IdentityLayer()

        if batch_norm_type is None:
            batch_norm_type = Normalization.NormType.NONE

        self._layers = nn.Sequential(
            OrderedDict(
                [
                    ("first_layer", nn.Linear(input_size, block_width)),
                    ("activation", activation()),
                    ("projection", nn.Linear(block_width, input_size)),
                    ("dropout", dropout_layer),
                    ("norm", Normalization(batch_norm_type, input_size)),
                ]
            )
        )

        nn.init.xavier_uniform_(self._layers.first_layer.weight)
        nn.init.xavier_uniform_(self._layers.projection.weight)

    def forward(self, x):
        return x + self._layers(x)

    def get_input_size(self):
        return self._layer.first_layer.in_features

    def get_output_size(self):
        return self._layer.first_layer.out_features
