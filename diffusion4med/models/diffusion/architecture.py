from torch import nn
from torch import Tensor
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
from functools import partial


class FPN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        conv_layer: nn.Module,
        time_2_embs_layer: nn.Module,
        channels: tuple = (1, 2, 4, 8),
        num_groups: int = 4,
        timesteps: int = 300,
    ) -> None:
        super().__init__()

        self.init_conv = conv_layer(in_channels, channels[0], kernel_size=1)

        time_emb_dim = channels[-1]
        self.time_2_embs = time_2_embs_layer(timesteps, time_emb_dim)

        block = partial(
            ResBlock,
            num_groups=num_groups,
            conv_layer=conv_layer,
            time_emb_dim=time_emb_dim,
        )
        self.downpath = nn.ModuleList([])
        self.uppath = nn.ModuleList([])

        for in_channels, out_channels in zip(channels[:-1], channels[1:]):
            self.downpath.append(
                nn.ModuleList(
                    [
                        block(in_channels, in_channels),
                        block(in_channels, in_channels),
                        Residual(
                            PreNorm(
                                LinearAttention(in_channels, conv_layer=conv_layer),
                                in_channels,
                            )
                        ),
                        Downsample(in_channels, out_channels, conv_layer=conv_layer),
                    ]
                )
            )

        mid_channels = channels[-1]
        self.midpath = nn.Sequential(
            block(mid_channels, mid_channels),
            QuadraticAttention(mid_channels, conv_layer=conv_layer),
            block(mid_channels, mid_channels),
        )

        reversed_channels = list(reversed(channels))
        for in_channels, out_channels in zip(
            reversed_channels[:-1], reversed_channels[1:]
        ):
            self.uppath.append(
                nn.ModuleList(
                    [
                        Upsample(in_channels, out_channels, conv_layer=conv_layer),
                        block(2 * out_channels, out_channels),
                        block(2 * out_channels, out_channels),
                        Residual(
                            PreNorm(
                                LinearAttention(out_channels, conv_layer=conv_layer),
                                out_channels,
                            )
                        ),
                    ]
                )
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

        feature_pyramid = [image]

        for upsmaple, block1, block2, attention in self.uppath:
            image = upsmaple(image)

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


class UnionArchitecture(nn.Module):
    def __init__(self, backbone: FPN, head: HeadFPN) -> None:
        super().__init__()
        self.backbone = backbone()
        self.head = head()

    def forward(self, image: Tensor, time: Tensor):
        return self.head(self.backbone(image, time))


# 2D not implemented yet
FPN3d = partial(
    FPN,
    conv_layer=WeightStandardizedConv3d,
    time_2_embs_layer=LearnablePositionEmbeddings,
)

HeadFPN3d = partial(
    HeadFPN,
    conv_layer=WeightStandardizedConv3d,
)
