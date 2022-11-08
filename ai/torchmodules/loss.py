import torch
import torch.nn as nn


def nth_root(tens, root=2, epsilon=1e-7):
    return torch.pow(tens + epsilon, 1 / root)


class CosineLoss(nn.Module):
    def __init__(self, root=2, use_scale=False) -> None:
        super().__init__()
        self._root = root
        self._use_scale = use_scale

    def forward(self, output, y_target) -> torch.Tensor:
        output_norm_val = torch.norm(output, dim=-1)
        y_target_norm_val = torch.norm(y_target, dim=-1)

        output_norm = output / torch.norm(output, dim=-1, keepdim=True)
        y_target_norm = y_target / torch.norm(y_target, dim=-1, keepdim=True)
        cosine_diffs = (1 - torch.sum(output_norm * y_target_norm, axis=1)) / 2
        exp_loss = nth_root(cosine_diffs, self._root)

        if self._use_scale:
            scale_diff = 1 + torch.abs(y_target_norm_val - output_norm_val) / y_target_norm_val
            exp_loss = scale_diff * exp_loss

        return exp_loss.mean()


def cosine_loss(output, y_target, root=2, use_scale=False):
    output_norm_val = torch.norm(output, dim=-1)
    y_target_norm_val = torch.norm(y_target, dim=-1)

    output_norm = output / torch.norm(output, dim=-1, keepdim=True)
    y_target_norm = y_target / torch.norm(y_target, dim=-1, keepdim=True)
    cosine_diffs = (1 - torch.sum(output_norm * y_target_norm, axis=1)) / 2
    exp_loss = nth_root(cosine_diffs, root)

    if use_scale:
        scale_diff = 1 + torch.abs(y_target_norm_val - output_norm_val) / y_target_norm_val
        exp_loss = scale_diff * exp_loss

    return exp_loss.mean()
