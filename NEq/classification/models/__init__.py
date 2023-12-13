from torchvision.models import (
    resnet18,
    resnet50,
    mobilenet_v2,
    ResNet18_Weights,
    ResNet50_Weights,
    MobileNet_V2_Weights,
)
import torch.nn as nn

from core.utils.config import config


def get_model():
    net_name = config.net_config.net_name
    print(f"Initialize model {net_name}")

    if net_name == "resnet18":
        model = resnet18()
    elif net_name == "resnet50":
        model = resnet50()
    elif net_name == "mbv2":
        model = mobilenet_v2()

    elif net_name == "pre_trained_resnet18":
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
    elif net_name == "pre_trained_resnet50":
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
    elif net_name == "pre_trained_mbv2":
        model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

    else:
        raise ValueError(f"No such model {config.net_config.net_name}")

    total_neurons = 0

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            total_neurons += m.weight.shape[0]

    return model, total_neurons
