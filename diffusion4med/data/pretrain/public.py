from amid import AMOS, FLARE2022, NLST, LIDC, NSCLC, MIDRC
from torch.utils.data import Dataset
from connectome import Chain, Filter, Apply, GroupBy, Transform, Merge
from pathlib import Path
import numpy as np
from tqdm import tqdm
from vox2vec.processing import (
    LocationsToSpacing, FlipAxesToCanonical, CropToBox, RescaleToSpacing,
    get_body_mask, BODY_THRESHOLD_HU, sample_box, scale_hu
)
from vox2vec.utils.box import mask_to_bbox
from vox2vec.utils.misc import is_diagonal
from imops import crop_to_box


PathLike = Path | str


def _prepare_nlst_ids(nlst_dir, patch_size):
    nlst = NLST(root=nlst_dir)
    for id_ in tqdm(nlst.ids, desc='Warming up NLST().patient_id method'):
        nlst.patient_id(id_)

    nlst_patients = nlst >> GroupBy('patient_id')
    ids = []
    for patient_id in tqdm(nlst_patients.ids, desc='Preparing NLST ids'):
        id, slice_locations = max(nlst_patients.slice_locations(patient_id).items(), key=lambda i: len(i[1]))
        if len(slice_locations) >= patch_size[2]:
            ids.append(id)

    return ids

class Public(Dataset):
    def __init__(
            self,
            spacing: tuple[float, float, float],
            patch_size: tuple[int, int, int],
            window_hu: tuple[float, float],
            amos_dir: PathLike | None = None,
            flare_dir: PathLike | None = None,
            nlst_dir: PathLike | None = None,
            midrc_dir: PathLike | None = None,
            nsclc_dir: PathLike | None = None,
            cache_dir: PathLike | None = None,
    ) -> None:
        if cache_dir is not None:
            from connectome import CacheToDisk

            def cache_to_disk(names):
                return CacheToDisk.simple(*names, root=cache_dir)
        else:
            from amid import CacheToDisk as cache_to_disk

        parse_affine = Transform(
            __inherit__=True,
            flipped_axes=lambda affine: tuple(np.where(np.diag(affine[:3, :3]) < 0)[0] - 3),  # enumerate from the end
            spacing=lambda affine: tuple(np.abs(np.diag(affine[:3, :3]))),
        )

        amos_ct_ids = AMOS(root=amos_dir).ids[:500]
        amos = Chain(
            AMOS(root=amos_dir),
            Filter.keep(amos_ct_ids),
            parse_affine,
            FlipAxesToCanonical(),
        )

        flare = Chain(
            FLARE2022(root=flare_dir),
            Filter(lambda id: id.startswith('TU'), verbose=True),
            Filter(lambda affine: is_diagonal(affine[:3, :3]), verbose=True),
            cache_to_disk(['ids']),
            parse_affine,
            FlipAxesToCanonical(),
        )

        nlst = Chain(
            NLST(root=nlst_dir),
            Transform(__inherit__=True, ids=lambda: _prepare_nlst_ids(nlst_dir, patch_size)),
            cache_to_disk(['ids']),
            LocationsToSpacing(),
            Apply(image=lambda x: np.flip(x, axis=(0, 1)).copy())
        )

        midrc = Chain(
            MIDRC(root=midrc_dir),
            Apply(image=lambda x: np.flip(x, axis=(0, 1)).copy())
        )

        nsclc = Chain(
            NSCLC(root=nsclc_dir),
            Apply(image=lambda x: np.flip(x, axis=(0, 1)).copy())
        )

        lidc = Chain(
            LIDC(),  # see amid docs
            Apply(image=lambda x: np.flip(np.swapaxes(x, 0, 1), axis=(0, 1)).copy())
        )

        # use connectome for smart cashing (with automatic invalidation)
        pipeline = Chain(
            Merge(
                amos,  # 500 abdominal CTs
                flare,  # 2000 abdominal CTs
                nlst,  # ~2500 thoracic CTs
                midrc,  # ~150 thoracic CTs (most patients with COVID-19)
                nsclc,  # ~400 thoracic CTs (most patients with severe non-small cell lung cancer)
                lidc  # ~1000 thoracic CTs (most patients with lung nodules)
            ),  # ~6550 openly available CTs in total, covering abdomen and thorax domains
            # cache spacing
            cache_to_disk(['spacing']),
            Filter(lambda spacing: spacing[-1] is not None, verbose=True),
            cache_to_disk(['ids']),
            # cropping, rescaling
            Transform(__inherit__=True, cropping_box=lambda image: mask_to_bbox(image >= BODY_THRESHOLD_HU)),
            CropToBox(axis=(-3, -2, -1)),
            RescaleToSpacing(to_spacing=spacing, axis=(-3, -2, -1), image_fill_value=lambda x: np.min(x)),
            Apply(image=lambda x: np.int16(x)),
            cache_to_disk(['image']),
            Apply(image=lambda x: np.float32(x)),
            # filtering by shape
            Filter(lambda image: np.all(np.array(image.shape) >= patch_size), verbose=True),
            cache_to_disk(['ids']),
            # adding roi_voxels
            Transform(__inherit__=True, body_mask=lambda image: get_body_mask(image)),
            cache_to_disk(['body_mask']),
            Filter(lambda body_mask: body_mask.any(), verbose=True),
            cache_to_disk(['ids']),
        )

        self.pipeline = pipeline
        self.ids = pipeline.ids
        self.load_example = pipeline._compile(['image', 'body_mask'])
        self.spacing = spacing
        self.patch_size = patch_size
        self.window_hu = window_hu

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        image, body_mask = self.load_example(self.ids[i])

        box = sample_box(image.shape, self.patch_size)
        image, body_mask = crop_to_box(image, box), crop_to_box(body_mask, box)

        image = scale_hu(image, self.window_hu)

        image = image[None]  # add channel dim

        return image
        
