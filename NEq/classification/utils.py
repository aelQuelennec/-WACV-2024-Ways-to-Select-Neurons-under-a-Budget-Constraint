from core.optim import MaskedSGD, MaskedAdam
from core.utils.config import config


def get_optimizer(model):
    print(f"Initialize optimizer {config.run_config.optimizer_name}")

    # Define optimizer and scheduler
    named_params = list(map(list, zip(*list(model.named_parameters()))))
    if config.run_config.optimizer_name == "sgd":
        return MaskedSGD(
            named_params[1],
            names=named_params[0],
            lr=config.run_config.base_lr,
            weight_decay=config.run_config.weight_decay,
            momentum=config.run_config.momentum,
        )
    if config.run_config.optimizer_name == "adam":
        return MaskedAdam(
            named_params[1],
            names=named_params[0],
            lr=config.run_config.base_lr,
            weight_decay=config.run_config.weight_decay,
        )
