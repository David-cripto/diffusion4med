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
        log_image_step: int = 10,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.timesteps = timesteps
        self.scheduler = scheduler(timesteps)
        # works only for 3D yet
        self.image_shape = image_shape if len(image_shape) == 5 else (1, *image_shape)
        self.log_image_step = log_image_step

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

        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        self.register_buffer("variance", variance)

    def q_sample(self, image: Tensor, time: Tensor, noise: Tensor):
        return (
            extract(self.sqrt_alphas_cumprod, time, image.shape) * image
            + extract(self.sqrt_one_minus_alphas_cumprod, time, image.shape) * noise
        )

    @torch.no_grad()
    def one_back_step(self, x_t: Tensor, time: Tensor):
        inverse_sqrt_alpha_t = extract(self.inverse_sqrt_alphas, time, self.image_shape)
        beta_t = extract(self.betas, time, self.image_shape)
        inverse_sqrt_one_minus_alphas_cumprod_t = extract(
            self.inverse_sqrt_one_minus_alphas_cumprod, time, self.image_shape
        )

        x_mean = inverse_sqrt_alpha_t * (
            x_t - beta_t * self(x_t, time) * inverse_sqrt_one_minus_alphas_cumprod_t
        )
        variance_t = extract(self.variance, time, self.image_shape)
        noise = torch.randn_like(x_t, device=self.device)
        return x_mean + (1.0 - (time == 0).float()) * torch.sqrt(variance_t) * noise

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
        x_time = self.q_sample(batch, time, noise)
        return self.criterion(self(x_time, time), noise)
