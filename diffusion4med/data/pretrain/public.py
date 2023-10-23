# do actually need dependecies from vox2vec?
from imops import crop_to_box
from vox2vec.processing import scale_hu, sample_box
import numpy as np
import torch
from vox2vec.pretrain.data.augmentations import sample_patch_size, sample_positive_pairs
from vox2vec.pretrain.data.public import PublicPretrainDataset

class ModifiedPublic(PublicPretrainDataset):
    def __getitem__(self, i):
        image, spacing, body_mask = self.load_example(self.ids[i])
        spacing = np.array(spacing, dtype='float32')
                
        patch_size = sample_patch_size(image.shape, spacing, self.spatial_augmentations)
        positive_pairs = [
            sample_positive_pairs(image, spacing, body_mask, patch_size, self.spatial_augmentations,
                                  self.color_augmentations, self.masking, self.max_num_voxels_per_patch)
            for _ in range(self.batch_size)
        ]
        patches_1, masks_1, voxels_1, patches_2, masks_2, voxels_2, orig_locs_mm, rois = zip(*positive_pairs)
        patches_1 = torch.tensor(np.stack([p[None] for p in patches_1]))        
        
        patches_1 = patches_1 * 2 - 1

        return patches_1