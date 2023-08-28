from lightning import LightningModule
from torch import nn, Tensor
import torch
from diffusion4med.models.autoencoders.distributions import DiagonalGaussianDistribution


class AutoencoderKL(LightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        z_channels: int,
        loss: nn.Module,
        learning_rate: float,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        encoder_out_channels = encoder.out_channels
        decoder_in_channels = decoder.in_channels
        self.to_mean_variance = nn.Conv3d(
            encoder_out_channels, 2 * z_channels, kernel_size=1
        )
        self.to_z = nn.Conv3d(z_channels, decoder_in_channels, kernel_size=1)
        self.loss = loss
        self.learning_rate = learning_rate

    def encode(self, image: Tensor):
        z = self.encoder(image)
        mean_variance = self.to_mean_variance(z)
        posterior = DiagonalGaussianDistribution(mean_variance)
        return posterior

    def decode(self, z: Tensor):
        z = self.to_z(z)
        image = self.decoder(z)
        return image

    def forward(self, image: Tensor, sample_posterior: bool = True):
        posterior = self.encode(image)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def training_step(self, batch: Tensor, batch_idx: int):
        reconstructions, posterior = self(batch)

        aeloss, log_dict_ae = self.loss(
            reconstructions, batch, posterior, split="train"
        )
        self.log(
            "aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True
        )
        self.log_dict(
            log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False
        )
        return aeloss

    def validation_step(self, batch: Tensor, batch_idx: int):
        reconstructions, posterior = self(batch)
        aeloss, log_dict_ae = self.loss(reconstructions, batch, posterior, split="val")
        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(
            log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False
        )
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.to_mean_variance.parameters())
            + list(self.to_z.parameters()),
            lr=lr,
            betas=(0.5, 0.9),
        )
        return [opt_ae], []
