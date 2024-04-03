import os
import yaml
import argparse
from easydict import EasyDict
from typing import Union

config = EasyDict()


def load_transfer_config(file_path: str) -> None:
    file_path = f"NEq/configs/{file_path}"

    def _iterative_update(dict1, dict2):
        for k in dict2:
            if k not in dict1:
                dict1[k] = dict2[k]
            else:  # k both in dict1 and dict2
                if isinstance(dict2[k], (dict, EasyDict)):
                    assert isinstance(dict1[k], (dict, EasyDict))
                    _iterative_update(dict1[k], dict2[k])
                else:
                    dict1[k] = dict2[k]

    global config
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            _config = yaml.safe_load(f)
            _iterative_update(config, _config)


def update_config_from_args(args: Union[dict, argparse.Namespace]):
    global config

    def _iterative_update(edict, new_k, new_v):
        for _k, _v in edict.items():
            if _k == new_k:
                edict[new_k] = new_v
                return True
        for _, _v in edict.items():
            if isinstance(_v, (dict, EasyDict)):
                _ret = _iterative_update(_v, new_k, new_v)
                if _ret:
                    return True
        return False

    if isinstance(args, argparse.Namespace):
        args = args.__dict__
    for k, v in args.items():
        if v is None or k == "config":
            continue
        ret = _iterative_update(config, k, v)
        if not ret:
            raise ValueError(f"ERROR: Updating args failed: cannot find key: {k}")


def parse_unknown_args(unknown):
    def _convert_value(_v):
        try:  # int
            return int(_v)
        except ValueError:
            pass
        try:  # float
            return float(_v)
        except ValueError:
            pass
        return _v  # string

    assert len(unknown) % 2 == 0
    parsed = dict()
    for idx in range(len(unknown) // 2):
        k, v = unknown[idx * 2], unknown[idx * 2 + 1]
        assert k.startswith("--")
        k = k[2:]
        v = _convert_value(v)
        parsed[k] = v
    return parsed


def update_config_from_unknown_args(unknown):
    parsed = parse_unknown_args(unknown)
    print(" * Getting extra args", parsed)
    update_config_from_args(parsed)


# iteratively convert easy dict to dictionary
def configs2dict(cfg):
    from easydict import EasyDict

    if isinstance(cfg, EasyDict):
        cfg = dict(cfg)
        key2cast = [k for k in cfg if isinstance(cfg[k], EasyDict)]
        for k in key2cast:
            cfg[k] = configs2dict(cfg[k])
        return cfg
    else:
        return cfg


# This function updates the configuration in case of sweep with the sweep parameters
def update_config_from_wandb(wandb_config):
    file_path = "NEq_configs.yaml"

    def _iterative_update(dict1, dict2):
        for k in dict2:
            if k not in dict1:
                dict1[k] = dict2[k]
            else:  # k both in dict1 and dict2
                if isinstance(dict2[k], (dict, EasyDict)):
                    assert isinstance(dict1[k], (dict, EasyDict))
                    _iterative_update(dict1[k], dict2[k])
                else:
                    dict1[k] = dict2[k]

    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    # Read NEq_configs.yaml to wandb_sweeps_config
    wandb_sweeps_config = EasyDict()
    with open(file_path, "r") as f:
        _config = yaml.safe_load(f)
        _iterative_update(wandb_sweeps_config, _config)

    config.manual_seed = wandb_config.manual_seed  # Read from sweep

    # Read dataset config
    dataset_config = wandb_sweeps_config.dataset_configs[wandb_config.dataset]
    config.data_provider.dataset = dataset_config.dataset
    config.data_provider.root = dataset_config.root
    config.data_provider.new_num_classes = dataset_config.new_num_classes

    config.NEq_config.neuron_selection = (
        wandb_config.neuron_selection
    )  # Read from sweep
    config.NEq_config.initialization = wandb_config.initialization  # Read from sweep
    config.net_config.net_name = wandb_config.net_name  # Read from sweep

    config.NEq_config.total_num_params = wandb_sweeps_config.networks[
        wandb_config.net_name
    ].total_num_params
    if (
        wandb_config.scheme == "scheme_fixed_budget"
        or "mcunet" in wandb_config.scheme
        or "proxyless" in wandb_config.scheme
        or "mbv2-w0.35" in wandb_config.scheme
    ):
        if (
            "budget" in wandb_config
        ):  # If this parameter in sweep file, update from sweep.
            config.NEq_config.glob_num_params = wandb_config.budget  # Read from sweep
        else:  # Otherwise, update from NEq_configs.yaml
            config.NEq_config.glob_num_params = wandb_sweeps_config.net_configs[
                wandb_config.scheme
            ].budget
    else:  # Otherwise
        config.NEq_config.ratio = wandb_sweeps_config.net_configs[
            wandb_config.scheme
        ].ratio
        config.NEq_config.glob_num_params = (
            config.NEq_config.total_num_params * config.NEq_config.ratio
        )

    if "n_epochs" in wandb_config:
        config.run_config.n_epochs = (
            wandb_config.n_epochs
        )  # Read from sweep, otherwise, it will be defined by transfer.yaml
    if "base_lr" in wandb_config:
        config.run_config.base_lr = (
            wandb_config.base_lr
        )  # Read from sweep, otherwise, it will be defined by transfer.yaml
    if "lr_schedule_name" in wandb_config:
        config.run_config.lr_schedule_name = (
            wandb_config.lr_schedule_name
        )  # Read from sweep, it will be defined by transfer.yaml
    if "optimizer_name" in wandb_config:
        config.run_config.optimizer_name = (
            wandb_config.optimizer_name
        )  # Read from sweep, it will be defined by transfer.yaml
    if "image_size" in wandb_config:
        config.data_provider.image_size = (
            wandb_config.image_size
        )  # Read from sweep, it will be defined by transfer.yaml
    if "base_batch_size" in wandb_config:
        config.data_provider.base_batch_size = (
            wandb_config.base_batch_size
        )  # Read from sweep, it will be defined by transfer.yaml
    if "use_validation" in wandb_config:
        config.data_provider.use_validation = (
            wandb_config.use_validation
        )  # Read from sweep, it will be defined by transfer.yaml
    if config.NEq_config.neuron_selection == "velocity":
        config.data_provider.use_validation_for_velocity = 1
    else:
        config.data_provider.use_validation_for_velocity = 0
    # config.data_provider.use_validation_for_velocity = 1

    if wandb_config.scheme == "scheme_fixed_budget" and (
        config.NEq_config.initialization == "SU"
        or config.NEq_config.neuron_selection == "SU"
    ):
        print("!! Warning, SU update can not apply with fixed budget scheme !!")

    if (
        config.NEq_config.initialization == "SU"
        or config.NEq_config.neuron_selection == "SU"
    ):
        backward_config = wandb_sweeps_config.net_configs[wandb_config.scheme].SU_scheme
        config.backward_config.n_bias_update = backward_config.n_bias_update
        config.backward_config.weight_update_ratio = backward_config.weight_update_ratio
        config.backward_config.manual_weight_idx = backward_config.manual_weight_idx
    if wandb_config.scheme == "scheme_fixed_budget":
        config.run_dir = f"runs/{wandb_config.dataset}/{wandb_config.net_name}/{wandb_config.initialization}/{wandb_config.neuron_selection}/{wandb_config.scheme}/{config.NEq_config.budget}/{wandb_config.manual_seed}"
    else:
        config.run_dir = f"runs/{wandb_config.dataset}/{wandb_config.net_name}/{wandb_config.initialization}/{wandb_config.neuron_selection}/{wandb_config.scheme}/{wandb_config.manual_seed}"
