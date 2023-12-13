import math
from core.utils.config import config
from torchvision import transforms

__all__ = ["ImageTransform"]


class ImageTransform(dict):
    def __init__(self):
        super().__init__(
            {
                "train": self.build_train_transform(),
                "val": self.build_val_transform(),
            }
        )

    def build_train_transform(self):
        if config.data_provider.dataset == "vww":
            t = transforms.Compose(
                [
                    transforms.Resize(
                        (
                            config.data_provider.image_size,
                            config.data_provider.image_size,
                        )
                    ),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(**self.mean_std),
                ]
            )
        else:
            from timm.data import create_transform

            t = create_transform(
                input_size=config.data_provider.image_size,
                is_training=True,
                color_jitter=config.data_provider.color_aug,
                mean=self.mean_std["mean"],
                std=self.mean_std["std"],
            )

        return t

    def build_val_transform(self):
        if config.data_provider.dataset == "vww":
            t = transforms.Compose(
                [
                    transforms.Resize(
                        (
                            config.data_provider.image_size,
                            config.data_provider.image_size,
                        )
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(**self.mean_std),
                ]
            )
        else:
            t = transforms.Compose(
                [
                    transforms.Resize(
                        int(math.ceil(config.data_provider.image_size / 0.875))
                    ),
                    transforms.CenterCrop(config.data_provider.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(**self.mean_std),
                ]
            )
        return t

    @property
    def mean_std(self):
        return config.data_provider.get(
            "mean_std", {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}
        )
