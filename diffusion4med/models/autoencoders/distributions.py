import torch


class DiagonalGaussianDistribution(object):
    def __init__(self, mean_variance, deterministic=False):
        self.mean_variance = mean_variance
        self.mean, self.logvar = torch.chunk(mean_variance, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(
                self.mean, device=self.mean_variance.device
            )

    def sample(self):
        x = self.mean + self.std * torch.randn(
            self.mean.shape, device=self.mean_variance.device
        )
        return x

    def mode(self):
        return self.mean

    def kl(self):
        if self.deterministic:
            return torch.tensor([0.0], device=self.mean_variance.device)
        else:
            return 0.5 * torch.sum(
                torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                dim=[-4, -3, -2, -1],
            )
