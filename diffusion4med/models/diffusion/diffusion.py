from thunder import ThunderModule
import torch
from diffusion4med.models.diffusion.utils import extract


class Diffusion(ThunderModule):
    def __init__(self, timesteps, scheduler, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.timesteps = timesteps
        self.scheduler = scheduler

        self.register_schedule()

    def register_schedule(self):
        betas = self.scheduler.get_betas()
        alphas = 1.0 - betas
        alphas_cum_prod = torch.cumprod(alphas, dim=0)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cum_prod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cum_prod)
        self.register_buffer("sqrt_alphas_cumprod", sqrt_alphas_cumprod)
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", sqrt_one_minus_alphas_cumprod
        )

    def q_sample(self, image, time, noise):
        return (
            extract(self.sqrt_alphas_cumprod * image, time, image.shape)
            + extract(self.sqrt_one_minus_alphas_cumprod, time, image.shape) * noise
        )

    def training_step(self, batch, batch_index):
        time = torch.randint(0, self.timesteps, size=(batch.shape[0],))
        noise = torch.randn_like(batch)
        x_time = self.q_sample(batch, time, noise)
        return self.criterion(self(x_time, time), noise)
