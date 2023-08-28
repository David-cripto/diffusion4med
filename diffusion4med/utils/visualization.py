from torch import Tensor
import matplotlib.pyplot as plt
import torchvision
import numpy as np
from lightning import Callback, LightningModule, Trainer
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
        axs[0, i].imshow(np.asarray(img), cmap="gray")
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    return fig


class ValVisualization(Callback):
    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        images = pl_module.generation()
        images_tensor = torch.vstack(images)
        images_slice = images_tensor[..., pl_module.slice_visualize]
        images_slice_normalize = [
            torch.from_numpy(bytescale(images_slice[i].cpu().numpy()))
            for i in range(images_slice.shape[0])
        ]
        images = [wandb.Image(show(img)) for img in images_slice_normalize]
        pl_module.logger.log_image(key=f"val/image_generating_process", images=images)

        return super().on_validation_epoch_end(trainer, pl_module)
