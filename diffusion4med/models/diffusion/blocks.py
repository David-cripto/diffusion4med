from torch import Tensor
from torch.nn import Conv2d
from einops import reduce
from functools import partial
import torch
from torch import nn
import torch.nn.functional as F


class WeightStandardizedConv2d(Conv2d):
    def forward(self, x: Tensor) -> Tensor:
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        mean = reduce(self.weight, "o ... -> o 1 1 1", "mean")
        var = reduce(self.weight, "o ... -> o 1 1 1", torch.var)

        normalized_weight = (self.weight - mean) / torch.sqrt(var + eps)
        return F.conv2d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class Block(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, num_groups: int = 8
    ) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            WeightStandardizedConv2d(in_channels, out_channels),
            nn.GroupNorm(num_groups, out_channels),
        )
        self.act = nn.SiLU()

    def forward(self, image, time_scale_shift=None):
        image = self.projection(image)
        if time_scale_shift:
            scale, shift = time_scale_shift
            image = image * (scale + 1) + shift

        return self.act(image)


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_groups: int = 8,
        time_emb_dim: int | None = None,
    ) -> None:
        super().__init__()

        self.wtime_block = Block(
            in_channels=in_channels, out_channels=out_channels, num_groups=num_groups
        )

        self.wo_time_block = Block(
            in_channels=out_channels, out_channels=out_channels, num_groups=num_groups
        )

        self.time_expansion = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_channels * 2))
            if time_emb_dim
            else None
        )
        self.identity = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, image, time=None):
        time_scale_shift = None

        if time and self.time_expansion:
            expanded_time = self.time_expansion(time)
            time_scale_shift = expanded_time.chunk(2, dim=1)

        new_image = self.wtime_block(image, time_scale_shift)
        new_image = self.wo_time_block(new_image)

        return new_image + self.identity(image)
