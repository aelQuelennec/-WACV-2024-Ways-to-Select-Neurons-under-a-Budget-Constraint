import torchvision
import pyvww
import torch
import torch.utils.data as data_utils
from typing import Tuple

from .vision import *
from ..utils.config import config
from .vision.transform import *

__all__ = ["build_dataset"]


class MapDataset(data_utils.Dataset):
    """Given a dataset, creates a dataset which applies a mapping function to its items (lazily, only when an item is called).

    Note that data is not cloned/copied from the initial dataset.

    Args:
        dataset:
        map_fn:
    """

    def __init__(self, dataset, map_fn, with_target=False):
        self.dataset = dataset
        self.map = map_fn
        self.with_target = with_target

    def __getitem__(self, index):
        if self.with_target:
            return self.map(self.dataset[index][0], self.dataset[index][1])
        else:
            return self.map(self.dataset[index][0]), self.dataset[index][1]

    def __len__(self):
        return len(self.dataset)


def split_dataset(
    dataset: torch.utils.data.Dataset, val_len: int = 10
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Randomly splits a `torch.utils.data.Dataset` instance in two non-overlapping separated `Datasets`.

    The split of the elements of the original `Dataset` is based on `val_len`, the length of the target validation dataset.

    Args:
        dataset (torch.utils.data.Dataset): `torch.utils.data.Dataset` instance to be split.
        val_len (float): number of elements of `dataset` contained in the second dataset.

    Returns:
        tuple: a tuple containing the two new datasets.

    """
    dataset_length = int(len(dataset))
    train_length = int(dataset_length - val_len)
    train_dataset, valid_dataset = data_utils.random_split(
        dataset,
        [train_length, val_len],
        generator=torch.Generator().manual_seed(config.manual_seed),
    )

    return train_dataset, valid_dataset


def build_dataset():
    if config.data_provider.dataset == "image_folder":
        train_dataset, test = image_folder(
            root=config.data_provider.root,
            transforms=ImageTransform(),
        )

    #Flowers 102 has its own validation set
    elif config.data_provider.dataset == "flowers102":
        train, validation_set, test = FLOWERS102(
            root=config.data_provider.root,
            transforms=ImageTransform(),
        )
        validation, validation_for_velocity = split_dataset(dataset=validation_set) # Take 10 elements from val set
        return {"train": train, "val": validation, "test": test, "val_velocity": validation_for_velocity}

    elif config.data_provider.dataset == "cifar10":
        train_dataset = torchvision.datasets.CIFAR10(
            config.data_provider.root,
            train=True,
            transform=None,
            download=True,
            )
        test = torchvision.datasets.CIFAR10(
                config.data_provider.root,
                train=False,
                transform=ImageTransform()["val"],
                download=True,
            )

    elif config.data_provider.dataset == "cifar100":
        train_dataset = torchvision.datasets.CIFAR100(
            config.data_provider.root,
            train=True,
            transform=None,
            download=True,
        )
        test = torchvision.datasets.CIFAR100(
            config.data_provider.root,
            train=False,
            transform=ImageTransform()["val"],
            download=True,
        )

    elif config.data_provider.dataset == "vww":
        train_dataset = pyvww.pytorch.VisualWakeWordsClassification(
            root="./data/vww/all2014",
            annFile="./data/vww/annotations/instances_train2014.json",
            transform=None,
        )
        test = pyvww.pytorch.VisualWakeWordsClassification(
            root="./data/vww/all2014",
            annFile="./data/vww/annotations/instances_val2014.json",
            transform=ImageTransform()["val"],
        )

    else:
        raise NotImplementedError(config.data_provider.dataset)

    # These operations allows for the creation of a small validation dataset from which to compute velocities
    train, validation_set = split_dataset(dataset = train_dataset, val_len = int(config.data_provider.validation_percentage * len(train_dataset))) #Divide the train_dataset into train and validation according to a predifined validation_percentage
    validation, validation_for_velocity = split_dataset(dataset=validation_set) # Take 10 elements from val set

    train = MapDataset(train, ImageTransform()["train"])
    validation = MapDataset(validation, ImageTransform()["val"])
    validation_for_velocity = MapDataset(validation_for_velocity, ImageTransform()["val"])

    return {"train": train, "val": validation, "test": test, "val_velocity": validation_for_velocity}
