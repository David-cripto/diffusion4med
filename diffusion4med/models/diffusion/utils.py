from typing import Any, Optional
from lightning import Callback, LightningModule, Trainer
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from torch import Tensor
from torchvision.utils import make_grid
import torch
from dpipe.im.preprocessing import bytescale
import wandb

def show(imgs: Tensor):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = torchvision.transforms.functional.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img), cmap='gray')
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    return fig

class ValVisulization(Callback):
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        images = pl_module.generation()
        images_tensor = torch.vstack(images)
        images_slice = images_tensor[..., 8]
        images_slice_normalize = [torch.from_numpy(bytescale(images_slice[i].cpu().numpy())) for i in range(images_slice.shape[0])]
        images = [wandb.Image(show(img)) for img in images_slice_normalize]
        pl_module.logger.log_image(key=f'val/image_generating_process', images=images)
        
        return super().on_validation_epoch_end(trainer, pl_module)

def exists(x):
    return x is not None


def extract(arr, time, shape):
    batch_size = time.shape[0]
    out = arr.gather(-1, time)
    return out.reshape(batch_size, *((1,) * (len(shape) - 1))).to(time.device)
