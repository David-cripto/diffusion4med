from torch import nn, Tensor
import torch


class Loss(nn.Module):
    def __init__(self, logvar_init: float = 0.0) -> None:
        super().__init__()
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

    def forward(self, predict_image: Tensor, image: Tensor, posteriors, split: str):
        abs_loss = torch.abs(predict_image.contiguous() - image.contiguous())
        nll_loss = abs_loss / torch.exp(self.logvar) + self.logvar

        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        loss = nll_loss + kl_loss
        log = {
            "{}/total_loss".format(split): loss.clone().detach().mean(),
            "{}/logvar".format(split): self.logvar.detach(),
            "{}/kl_loss".format(split): kl_loss.detach().mean(),
            "{}/nll_loss".format(split): nll_loss.detach().mean(),
        }
        return loss, log
