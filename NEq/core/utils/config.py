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

    assert "configs/" in file_path
    prefix = file_path.split("configs/")[0]
    file_path = file_path[len(prefix) :]

    levels = file_path.split("/")
    for i_level in range(len(levels)):
        cur_config_path = prefix + "/".join(levels[: i_level + 1])
        if os.path.exists(cur_config_path):
            with open(cur_config_path, "r") as f:
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

    wandb_sweeps_config = EasyDict()
    with open(file_path, "r") as f:
        _config = yaml.safe_load(f)
        _iterative_update(wandb_sweeps_config, _config)

    config.manual_seed = wandb_config.manual_seed

    dataset_config = wandb_sweeps_config.dataset_configs[wandb_config.dataset]
    config.data_provider.dataset = dataset_config.dataset
    config.data_provider.root = dataset_config.root
    config.data_provider.new_num_classes = dataset_config.new_num_classes

    config.NEq_config.ratio = wandb_sweeps_config.net_configs[wandb_config.scheme].ratio
    config.NEq_config.total_num_params = wandb_sweeps_config.networks[
        wandb_config.net_name
    ].total_num_params
    config.NEq_config.neuron_selection = wandb_config.neuron_selection
    config.NEq_config.initialization = wandb_config.initialization

    config.net_config.net_name = wandb_config.net_name
    if (
        config.NEq_config.initialization == "SU"
        or config.NEq_config.neuron_selection == "SU"
    ):
        backward_config = wandb_sweeps_config.net_configs[wandb_config.scheme].SU_scheme
        config.backward_config.n_bias_update = backward_config.n_bias_update
        config.backward_config.weight_update_ratio = backward_config.weight_update_ratio
        config.backward_config.manual_weight_idx = backward_config.manual_weight_idx

    config.run_dir = f"runs/{wandb_config.dataset}/{wandb_config.net_name}/{wandb_config.initialization}/{wandb_config.neuron_selection}/{wandb_config.scheme}/{wandb_config.manual_seed}"
