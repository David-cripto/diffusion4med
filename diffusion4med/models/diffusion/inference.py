from torch import nn, Tensor
import torch
from vox2vec.eval.models.probing import MulticlassProbing
from vox2vec.nn.functional import (
    compute_multiclass_segmentation_loss,
    eval_mode,
    compute_dice_score,
)
from vox2vec.eval.models.predict import multiclass_predict
from dpipe.torch import to_var
from diffusion4med.models.diffusion import Diffusion
from vox2vec.utils.misc import identity
from vox2vec.eval.models.visualize import draw
from monai.metrics import compute_hausdorff_distance 

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


class ProbingModified(MulticlassProbing):
    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        optimizer.zero_grad()

        images, rois, gt_masks = batch
        
        images = images*2 - 1

        with torch.no_grad(), eval_mode(self.backbone):
            backbone_outputs = self.backbone(images)

        for i, head in enumerate(self.heads):
            pred_logits = head(backbone_outputs)
            loss, logs = compute_multiclass_segmentation_loss(pred_logits, gt_masks, rois,
                                                              logs_prefix=f'train/head_{i}_')
            self.log_dict(logs)
            self.manual_backward(loss)

        optimizer.step()

    def validation_step(self, batch, batch_idx):
        image, roi, gt_mask = batch
        
        image = image*2 - 1
        
        
        gt_mask = gt_mask[1:]  # drop background mask
        for i, head in enumerate(self.heads):
            pred_probas = multiclass_predict(image, self.patch_size, self.backbone, head,
                                             self.device, roi, self.sw_batch_size)
            argmax = pred_probas.argmax(dim=0)
            pred_mask = torch.stack([argmax == i for i in range(1, pred_probas.shape[0])])

            dice_scores = compute_dice_score(pred_mask, gt_mask, reduce=identity)
            for j, dice_score in enumerate(dice_scores):
                self.log(f'val/head_{i}_dice_score_for_cls_{j}', dice_score, on_epoch=True, sync_dist=True)
            self.log(f'val/head_{i}_avg_dice_score', dice_scores.mean(), on_epoch=True, sync_dist=True)

            if self.draw:
                for dim in range(3):
                    log_image = draw(image, gt_mask, pred_mask, dim)
                    self.logger.log_image(
                        f'val/image_{batch_idx}_dim_{dim}',
                        log_image, self.trainer.current_epoch
                    )

    def test_step(self, batch, batch_idx):
        image, roi, gt_mask, spacing = batch
        
        image = image*2 - 1
        
        gt_mask = gt_mask[1:]  # drop background mask
        for i, head in enumerate(self.heads):
            pred_probas = multiclass_predict(image, self.patch_size, self.backbone, head,
                                             self.device, roi, self.sw_batch_size)
            argmax = pred_probas.argmax(dim=0)
            pred_mask = torch.stack([argmax == i for i in range(1, pred_probas.shape[0])])

            dice_scores = compute_dice_score(pred_mask, gt_mask, reduce=identity)
            hd_scores = compute_hausdorff_distance(
                pred_mask.unsqueeze(0), gt_mask.unsqueeze(0), include_background=True, spacing=tuple(map(float, spacing))
            ).squeeze(0)

            for j in range(len(dice_scores)):
                self.log(f'test/head_{i}_dice_score_for_cls_{j}', dice_scores[j], on_epoch=True)
                self.log(f'test/head_{i}_hd_score_for_cls_{j}', hd_scores[j], on_epoch=True)
            self.log(f'test/head_{i}_avg_dice_score', dice_scores.mean(), on_epoch=True)
            self.log(f'test/head_{i}_avg_hd_score', hd_scores.mean(), on_epoch=True)

            if self.draw:
                for dim in range(3):
                    for slc in range(0, image.shape[dim + 1], 10):
                        log_image = draw(image, gt_mask, pred_mask, dim, slc)
                        self.logger.log_image(f'test/image_{batch_idx}_dim_{dim}', log_image, slc)
