from torch.utils.data import Dataset
from pathlib import Path
from diffusion4med.data.toys_datasets.utils import get_train_test_mnist3D
from torchvision.transforms import Compose, ToTensor, Lambda
import numpy as np
import torch

PathLike = Path | str


class Mnist3d(Dataset):
    def __init__(
        self,
        path_to_data: PathLike,
        transform: Compose = Compose(
            [
                torch.from_numpy,
                Lambda(lambda x: x.float().reshape(16, 16, 16).unsqueeze(0)),
                Lambda(lambda t: (t * 2) - 1),
            ]
        ),
        train: bool = True
    ) -> None:
        super().__init__()
        x_train, x_test, y_train, y_test = get_train_test_mnist3D(path_to_data)
        if train:
            self.images = np.vstack([x_train])
        else:
            self.images = np.vstack([x_test])
            
        self.transform = transform

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index: int):
        return self.transform(self.images[index])
