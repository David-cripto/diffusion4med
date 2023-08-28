from torch import nn, Tensor
import torch
from vox2vec.eval.probing import Probing
from vox2vec.nn.functional import (
    compute_binary_segmentation_loss,
    eval_mode,
    compute_dice_score,
)
from vox2vec.eval.predict import predict
from dpipe.torch import to_var
from diffusion4med.models.diffusion import Diffusion


class Backbone(nn.Module):
    def __init__(
        self, diffusion: Diffusion, times: tuple[int], path_to_ckpt: str
    ) -> None:
        super().__init__()
        self.diffusion = diffusion
        self.times = times
        self.path_to_ckpt = path_to_ckpt
        self.init_from_ckpt()

    def init_from_ckpt(self):
        state_dict = torch.load(self.path_to_ckpt)
        model_weights = {
            key[len("backbone.") :]: value
            for key, value in state_dict["state_dict"].items()
            if "backbone" in key
        }
        self.diffusion.backbone.load_state_dict(model_weights)

    def forward(self, image: Tensor) -> tuple[Tensor, ...]:
        embeddings = []
        for t in self.times:
            time = torch.full((image.shape[0],), t, device=self.diffusion.device)
            noise = torch.randn_like(image, device=self.diffusion.device)
            x_t = self.diffusion.q_sample(image, time, noise)
            fpn_embeddings = self.diffusion.backbone(x_t, time)
            if not embeddings:
                embeddings.extend(fpn_embeddings)
                continue
            embeddings = [
                torch.cat((collected_features, features_t), dim=1)
                for collected_features, features_t in zip(embeddings, fpn_embeddings)
            ]
        return list(reversed(embeddings))


class ProbingModified(Probing):
    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        optimizer.zero_grad()

        images, rois, gt_masks = batch
        ### crutch
        images = images * 2 - 1
        ###
        with torch.no_grad(), eval_mode(self.backbone):
            backbone_outputs = self.backbone(images)

        for i, head in enumerate(self.heads):
            pred_logits = head(backbone_outputs)
            loss, logs = compute_binary_segmentation_loss(
                pred_logits, gt_masks, rois, logs_prefix=f"train/head_{i}_"
            )
            self.log_dict(logs)
            self.manual_backward(loss)

        optimizer.step()

    def _val_test_step(self, image, roi, gt_mask, stage):
        image = image * 2 - 1
        for i, head in enumerate(self.heads):
            pred_probas = predict(
                image, self.patch_size, self.backbone, head, self.device, roi
            )
            pred_mask = pred_probas >= self.threshold
            dice_scores = compute_dice_score(pred_mask, gt_mask, reduce=lambda x: x)
            for j, dice_score in enumerate(dice_scores):
                self.log(
                    f"{stage}/head_{i}_dice_score_for_cls_{j}",
                    dice_score,
                    on_epoch=True,
                )
            self.log(
                f"{stage}/head_{i}_avg_dice_score", dice_scores.mean(), on_epoch=True
            )
