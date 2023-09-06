from torch import Tensor, einsum
from torch.nn import Conv3d, Conv2d
from einops import reduce, rearrange
import torch
from torch import nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from abc import abstractmethod, ABC
import typing as tp
import math
from diffusion4med.utils import exists
import logging
from functools import partial


class WeightStandardizedConv3d(Conv3d):
    def forward(self, image: Tensor) -> Tensor:
        if self.weight.size()[1:] == torch.Size([1, 1, 1, 1]):
            return super().forward(image)
        eps = 1e-5 if image.dtype == torch.float32 else 1e-3
        
        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1 1 1", partial(torch.var, unbiased=False))

        normalized_weight = (weight - mean) * (var + eps).rsqrt()
        
        return F.conv3d(
            image,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class WeightStandardizedConv2d(Conv2d):
    def forward(self, image: Tensor) -> Tensor:
        if self.weight.size()[1:] == torch.Size([1, 1, 1]):
            return super().forward(image)
        eps = 1e-5 if image.dtype == torch.float32 else 1e-3
        mean = reduce(self.weight, "o ... -> o 1 1 1", "mean")
        std = reduce(self.weight, "o ... -> o 1 1 1", torch.std)

        normalized_weight = (self.weight - mean) / (std + eps)
        if torch.norm(normalized_weight).detach() >= 1e5:
            logging.warning(
                f"Something wrond with std in WeightStandardizedConv2d with size = {normalized_weight.size()}"
            )
        return F.conv2d(
            image,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class Block(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_groups: int = 8,
        conv_layer: nn.Module = WeightStandardizedConv3d,
    ) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            conv_layer(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, out_channels),
        )
        self.act = nn.SiLU()

    def forward(self, image: Tensor, time_scale_shift: Tensor | None = None):
        image = self.projection(image)
        if exists(time_scale_shift):
            scale, shift = time_scale_shift
            image = image * (scale + 1) + shift

        return self.act(image)


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_groups: int = 8,
        conv_layer: nn.Module = WeightStandardizedConv3d,
        time_emb_dim: int | None = None,
    ) -> None:
        super().__init__()

        self.wtime_block = Block(
            in_channels=in_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            conv_layer=conv_layer,
        )

        self.wo_time_block = Block(
            in_channels=out_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            conv_layer=conv_layer,
        )

        self.time_expansion = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_channels * 2))
            if exists(time_emb_dim)
            else None
        )
        self.identity_func = (
            conv_layer(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, image: Tensor, time: Tensor | None = None):
        time_scale_shift = None
        if exists(time) and exists(self.time_expansion):
            expanded_time = self.time_expansion(time)
            expanded_time = rearrange(expanded_time, "b c -> b c 1 1 1")
            time_scale_shift = expanded_time.chunk(2, dim=1)

        new_image = self.wtime_block(image, time_scale_shift)
        new_image = self.wo_time_block(new_image)

        return new_image + self.identity_func(image)


class Downsample(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_layer: nn.Module = WeightStandardizedConv3d,
    ) -> None:
        super().__init__()
        self.func = conv_layer(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, image: Tensor):
        return self.func(image)


class Upsample(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_layer: nn.Module = WeightStandardizedConv3d,
    ) -> None:
        super().__init__()
        self.func = nn.Sequential(
            nn.Upsample(scale_factor=2),
            conv_layer(in_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, image: Tensor):
        return self.func(image)


class AbstractAttentnion(nn.Module, ABC):
    def __init__(
        self,
        in_channels: int,
        num_heads: int = 4,
        dim_head: int = 32,
        conv_layer: nn.Module = WeightStandardizedConv3d,
    ) -> None:
        super().__init__()
        hidden_dim = num_heads * dim_head
        self.qkv = conv_layer(in_channels, 3 * hidden_dim, kernel_size=1, bias=False)
        self.num_heads = num_heads
        self.scale = dim_head**-0.5
        self.out_func = conv_layer(hidden_dim, in_channels, kernel_size=1)

    @abstractmethod
    def forward(self, image: Tensor):
        pass

    @staticmethod
    def smart_softmax(x: Tensor, dim: int):
        x = x - x.amax(dim=dim, keepdim=True).detach()
        return torch.softmax(x, dim=dim)

    def get_qkv(self, image: Tensor):
        qkv = self.qkv(image).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y z -> b h c (x y z)", h=self.num_heads),
            qkv,
        )

        return q, k, v

    def calculate_ans(self, out: Tensor, h: int, w: int, d: int):
        out = rearrange(out, "b h (x y z) c-> b (h c) x y z", x=h, y=w, z=d)
        return self.out_func(out)


class QuadraticAttention(AbstractAttentnion):
    def forward(self, image: Tensor):
        b, c, h, w, d = image.shape

        q, k, v = self.get_qkv(image)

        attention = einsum("b h c i, b h c j -> b h i j", q, k)
        attention = attention * self.scale
        attention = QuadraticAttention.smart_softmax(attention, dim=-1)
        out = einsum("b h i j, b h c j -> b h i c", attention, v)

        return self.calculate_ans(out, h, w, d)


class LinearAttention(AbstractAttentnion):
    def forward(self, image: Tensor):
        b, c, h, w, d = image.shape

        q, k, v = self.get_qkv(image)

        q = LinearAttention.smart_softmax(q, dim=-2)
        k = LinearAttention.smart_softmax(k, dim=-1)

        q = q * self.scale

        context = einsum("b h l i, b h m i -> b h l m", k, v)
        out = einsum("b h l n, b h l m -> b h n m", q, context)
        return self.calculate_ans(out, h, w, d)


class PreNorm(nn.Module):
    def __init__(self, func: tp.Callable, in_channels) -> None:
        super().__init__()
        self.func = func
        self.norm = nn.GroupNorm(1, in_channels)

    def forward(self, image: Tensor):
        return self.func(self.norm(image))


class Residual(nn.Module):
    def __init__(self, func: tp.Callable) -> None:
        super().__init__()
        self.func = func

    def forward(self, image: Tensor):
        return image + self.func(image)


class AbstractPositionalEmbeddings(nn.Module, ABC):
    def __init__(self, timesteps: int, dim: int):
        super().__init__()
        self.timesteps = timesteps
        self.dim = dim

    @abstractmethod
    def forward(self, time: Tensor):
        pass


class SinusoidalPositionEmbeddings(AbstractPositionalEmbeddings):
    def forward(self, time: Tensor):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class LearnablePositionEmbeddings(AbstractPositionalEmbeddings):
    def __init__(self, timesteps: int, dim: int):
        super().__init__(timesteps, dim)
        self.to_embs = nn.Sequential(
            nn.Embedding(timesteps, dim), nn.LayerNorm(dim), nn.Dropout(p=0.1)
        )

    def forward(self, time: Tensor):
        return self.to_embs(time)
