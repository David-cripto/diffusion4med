from torch import nn, Tensor
import torch
from diffusion4med.models.diffusion.blocks import (
    ResBlock,
    Residual,
    PreNorm,
    LinearAttention,
    QuadraticAttention,
    Downsample,
    Upsample,
    WeightStandardizedConv2d,
    WeightStandardizedConv3d,
    LearnablePositionEmbeddings,
)
from torch.nn import Conv3d
from functools import partial
import deepspeed
from fairscale.nn.checkpoint import checkpoint_wrapper


class FPN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        conv_layer: nn.Module,
        time_2_embs_layer: nn.Module,
        channels: tuple[int, ...] = (1, 2, 4, 8),
        num_groups: int = 4,
        timesteps: int = 300,
        num_blocks: tuple[int, ...] = ((2, 2), (2, 2), (2, 2)),
    ) -> None:
        super().__init__()

        time_emb_dim = channels[-1]
        self.time_2_embs = time_2_embs_layer(timesteps, time_emb_dim)

        block = partial(
            ResBlock,
            num_groups=num_groups,
            conv_layer=conv_layer,
            time_emb_dim=time_emb_dim,
        )

        self.init_conv = block(in_channels, channels[0])

        self.downpath = nn.ModuleList([])
        self.uppath = nn.ModuleList([])

        for num_idx, (in_channels, out_channels) in enumerate(
            zip(channels[:-1], channels[1:])
        ):
            self.downpath.extend(
                nn.ModuleList(
                    [
                        block(in_channels, in_channels),
                        block(in_channels, in_channels),
                        Residual(
                            PreNorm(
                                LinearAttention(in_channels, conv_layer=Conv3d),
                                in_channels,
                            )
                        ),
                        Downsample(in_channels, out_channels, conv_layer=Conv3d)
                        if block_idx == num_blocks[num_idx][0] - 1
                        else nn.Identity(),
                    ]
                )
                for block_idx in range(num_blocks[num_idx][0])
            )

        mid_channels = channels[-1]
        self.midpath = nn.Sequential(
            block(mid_channels, mid_channels),
            Residual(
                PreNorm(
                    QuadraticAttention(mid_channels, conv_layer=Conv3d),
                    in_channels=mid_channels,
                )
            ),
            block(mid_channels, mid_channels),
        )

        reversed_channels = list(reversed(channels))
        for num_idx, (in_channels, out_channels) in enumerate(
            zip(reversed_channels[:-1], reversed_channels[1:])
        ):
            self.uppath.extend(
                nn.ModuleList(
                    [
                        Upsample(in_channels, out_channels, conv_layer=Conv3d)
                        if block_idx == 0
                        else nn.Identity(),
                        block(2 * out_channels, out_channels),
                        block(2 * out_channels, out_channels),
                        Residual(
                            PreNorm(
                                LinearAttention(out_channels, conv_layer=Conv3d),
                                out_channels,
                            )
                        ),
                    ]
                )
                for block_idx in range(num_blocks[len(num_blocks) - 1 - num_idx][1])
            )

    def forward(self, image: Tensor, time: Tensor):
        image = self.init_conv(image)

        time = self.time_2_embs(time)
        skip_feature_maps = []

        for block1, block2, attention, downsample in self.downpath:
            image = block1(image, time)
            skip_feature_maps.append(image)

            image = block2(image, time)
            image = attention(image)
            skip_feature_maps.append(image)

            image = downsample(image)

        image = self.midpath(image)

        feature_pyramid = []

        for upsample, block1, block2, attention in self.uppath:
            if isinstance(upsample, Upsample):
                feature_pyramid.append(image)
                
            image = upsample(image)

            image = torch.cat((image, skip_feature_maps.pop()), dim=1)
            image = block1(image, time)

            image = torch.cat((image, skip_feature_maps.pop()), dim=1)
            image = block2(image, time)

            image = attention(image)

        feature_pyramid.append(image)
        return feature_pyramid


class HeadFPN(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, conv_layer: nn.Module
    ) -> None:
        super().__init__()
        self.func = conv_layer(in_channels, out_channels, kernel_size=1)

    def forward(self, images: list[Tensor]):
        return self.func(images[-1])


# 2D not implemented yet
WeightStandardizedFPN3d = partial(
    FPN,
    conv_layer=WeightStandardizedConv3d,
    time_2_embs_layer=LearnablePositionEmbeddings,
)

WeightStandardizedHeadFPN3d = partial(
    HeadFPN,
    conv_layer=WeightStandardizedConv3d,
)

FPN3d = partial(
    FPN,
    conv_layer=Conv3d,
    time_2_embs_layer=LearnablePositionEmbeddings,
)

HeadFPN3d = partial(
    HeadFPN,
    conv_layer=Conv3d,
)
