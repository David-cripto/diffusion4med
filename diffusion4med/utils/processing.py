from connectome import Transform, Mixin
import numpy as np
from dicom_csv import locations_to_spacing
from imops import crop_to_box

def check_diagonal(matrix: np.ndarray) -> bool:
    return np.allclose(matrix, np.diag(np.diag(matrix)))


class ParseAffine(Transform):
    __inherit__ = True

    def flipped_axes(affine: np.ndarray):
        return tuple(np.where(np.diag(affine[:3, :3]) < 0)[0] - 3)

    def spacing(affine: np.ndarray):
        return tuple(np.abs(np.diag(affine[:3, :3])))


class FlipImage(Transform):
    __inherit__ = True

    def image(image: np.ndarray, flipped_axes: tuple[int, ...]):
        if not flipped_axes:
            return image
        return np.flip(image, axis=flipped_axes).copy()
    
class LocationsToSpacing(Transform):
    __inherit__ = True

    def spacing(pixel_spacing: tuple[int, int], slice_locations: np.ndarray):
        return (*pixel_spacing, locations_to_spacing(slice_locations))
    