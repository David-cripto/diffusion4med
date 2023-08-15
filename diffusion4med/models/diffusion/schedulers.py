import torch
from abc import ABC, abstractmethod

class Scheduler(ABC):
    def __init__(self, timesteps: int) -> None:
        self.timesteps = timesteps
        
    @abstractmethod
    def get_betas(self):
        pass
        
class SigmoidScheduler(Scheduler):
    def get_betas(self):
        beta_start = 0.0001
        beta_end = 0.02
        betas = torch.linspace(-6, 6, self.timesteps)
        return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start    

class CosineSheduler(Scheduler):
    def get_betas(self, s=0.008):
        steps = self.timesteps + 1
        x = torch.linspace(0, self.timesteps, steps)
        alphas_cumprod = torch.cos(((x / self.timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
