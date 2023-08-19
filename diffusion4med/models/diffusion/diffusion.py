from typing import Tuple
from lightning.pytorch.utilities.types import STEP_OUTPUT
from thunder import ThunderModule
import torch
from torch import Tensor
from diffusion4med.models.diffusion.utils import extract
from diffusion4med.models.diffusion.schedulers import Scheduler
from tqdm import tqdm
import torch.nn.functional as F


class Diffusion(ThunderModule):
    def __init__(
        self,
        timesteps: int,
        scheduler: Scheduler,
        image_shape: tuple[int, ...],
        num_log_images: int = 10,
        log_time_step: int | None = None,
        slice_visualize: int | None = None,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.timesteps = timesteps
        self.scheduler = scheduler(timesteps)
        # works only for 3D yet
        self.image_shape = image_shape if len(image_shape) == 5 else (1, *image_shape)
        self.log_image_step = timesteps // num_log_images 
        self.slice_visualize = slice_visualize if slice_visualize is not None else image_shape[-1] // 2
        self.log_time_step = log_time_step

        self.register_schedule()

    def register_schedule(self):
        betas = self.scheduler.get_betas()
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.register_buffer("betas", betas)
        self.register_buffer("sqrt_alphas_cumprod", sqrt_alphas_cumprod)
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", sqrt_one_minus_alphas_cumprod
        )
        self.register_buffer("inverse_sqrt_alphas", 1.0 / torch.sqrt(alphas))
        self.register_buffer(
            "inverse_sqrt_one_minus_alphas_cumprod", 1.0 / sqrt_one_minus_alphas_cumprod
        )
        inverse_sqrt_alphas_cumprod = 1.0 / sqrt_alphas_cumprod
        self.register_buffer("inverse_sqrt_alphas_cumprod", inverse_sqrt_alphas_cumprod)
        self.register_buffer(
            "sqrt_inverse_alphas_cumprod_m1",
            inverse_sqrt_alphas_cumprod * sqrt_one_minus_alphas_cumprod,
        )

        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.register_buffer(
            "posterior_coef_x0",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_coef_xt",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        self.register_buffer("variance", variance)

    def q_sample(self, image: Tensor, time: Tensor, noise: Tensor):
        return (
            extract(self.sqrt_alphas_cumprod, time, image.shape) * image
            + extract(self.sqrt_one_minus_alphas_cumprod, time, image.shape) * noise
        )

    def get_x0_from_noise(self, x_t: Tensor, time: Tensor, noise: Tensor):
        return (
            extract(self.inverse_sqrt_alphas_cumprod, time, noise) * x_t
            - extract(self.sqrt_inverse_alphas_cumprod_m1, time, noise) * noise
        ).clamp_(-1.0, 1.0)

    @torch.no_grad()
    def one_back_step(self, x_t: Tensor, time: Tensor):
        noise = self(x_t, time)
        x_0 = self.get_x0_from_noise(x_t=x_t, time=time, noise=noise)
        x_t = (
            extract(self.posterior_coef_x0, time, self.image_shape) * x_0
            + extract(self.posterior_coef_xt, time, self.image_shape) * x_t
        )

        variance_t = extract(self.variance, time, self.image_shape)
        noise = torch.randn_like(x_t, device=self.device)
        return x_t + (1.0 - (time == 0).float()) * torch.sqrt(variance_t) * noise

    @torch.no_grad()
    def generation(self):
        image = torch.randn(self.image_shape, device=self.device)
        log_images = [image]

        for t in tqdm(
            reversed(range(0, self.timesteps)),
            desc="Sampling",
            total=self.timesteps,
        ):
            image = self.one_back_step(
                x_t=image,
                time=torch.full((self.image_shape[0],), t, device=self.device),
            )
            if t % self.log_image_step == 0 or t == 0 or t == self.timesteps - 1:
                log_images.append(image)
        return log_images

    def training_step(self, batch: Tensor, batch_index: int):
        time = torch.randint(
            0, self.timesteps, size=(batch.shape[0],), device=self.device
        )
        noise = torch.randn_like(batch, device=self.device)
        x_t = self.q_sample(batch, time, noise)
        loss = self.criterion(self(x_t, time), noise)
        return loss

    def validation_step(
        self, batch: Tensor, batch_idx: int, dataloader_idx: int = 0
    ) -> STEP_OUTPUT:
        batch = batch.to(self.device)
        if self.log_time_step:
            time = torch.full(
                size=(batch.shape[0],),
                fill_value=self.log_time_step,
                device=self.device,
            )
        else:
            time = torch.full(
                size=(batch.shape[0],),
                fill_value=self.timesteps // 2,
                device=self.device,
            )
        noise = torch.randn_like(batch, device=self.device)
        x_t = self.q_sample(batch, time, noise)
        predicted_noise = self(x_t, time)
        x_0 = self.get_x0_from_noise(x_t=x_t, time=time, noise=predicted_noise)
        self.logger.log_image(
            "val/true_image | generated_image",
            images=[batch[..., self.slice_visualize], x_0[..., self.slice_visualize]],
            step=self.current_epoch,
        )
        return predicted_noise.cpu(), noise.cpu()
