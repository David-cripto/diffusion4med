import h5py
from pathlib import Path
import numpy as np

PathLike = Path | str


def get_train_test_mnist3D(path_to_data: PathLike) -> tuple[np.ndarray, ...]:
    with h5py.File(path_to_data, "r") as dataset:
        x_train = dataset["X_train"][:]
        x_test = dataset["X_test"][:]
        y_train = dataset["y_train"][:]
        y_test = dataset["y_test"][:]
    return x_train, x_test, y_train, y_test
