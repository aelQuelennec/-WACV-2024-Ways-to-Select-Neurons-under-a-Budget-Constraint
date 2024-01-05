import os
from typing import Callable, Dict, Optional

from torchvision import datasets
import warnings


class ImageFolderFilterWarning(datasets.ImageFolder):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader=datasets.folder.default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super().__init__(root, transform, target_transform, loader, is_valid_file)

    def __getitem__(self, index):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            return super().__getitem__(index)


def image_folder(root, transforms):
    train = ImageFolderFilterWarning(
        root=os.path.join(root, "train"),
        transform=None,
        target_transform=None,
    )
    test = ImageFolderFilterWarning(
        root=os.path.join(root, "val"),
        transform=transforms["val"],
        target_transform=None,
    )
    return train, test
