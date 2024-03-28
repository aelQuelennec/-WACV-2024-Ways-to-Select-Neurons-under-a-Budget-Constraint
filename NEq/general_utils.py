import os
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.utils.data
from torch import nn

from classification.models import get_model
from core.utils.config import config, load_transfer_config

from core.utils.sparse_update_tools import get_all_conv_ops


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "12356")

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def change_classifier_head(classifier):
    classifier = nn.Linear(
        in_features=classifier.in_features,
        out_features=config.data_provider.new_num_classes,
        bias=True,
    )


def count_net_num_conv_params(model):
    conv_ops = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
    num_params = []
    for conv in conv_ops:
        this_num_weight = 0
        if conv.bias is not None:
            this_num_weight += conv.bias.numel()
        this_num_weight += conv.weight.numel()
        num_params.append(this_num_weight)
    total_num_params = sum(num_params)
    print("Total num param: ", total_num_params)

def count_updateable_param(model, scheme): # Use to calculate the total updatable parameter of SU scheme

    ## Same as count_net_num_conv_params
    conv_ops = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
    num_params = []
    for conv in conv_ops:
        this_num_weight = 0
        if conv.bias is not None:
            this_num_weight += conv.bias.numel()
        this_num_weight += conv.weight.numel()
        num_params.append(this_num_weight)
    ## 
    import yaml

    with open("./NEq_configs.yaml", 'r') as file:
        NEq_configs_yaml = yaml.safe_load(file)
    # backward_config  = NEq_configs_yaml["net_configs"][sweep_config["scheme"][0]]["SU_scheme"]
    backward_config  = NEq_configs_yaml["net_configs"][scheme]["SU_scheme"]
    

    weight_update_ratio = backward_config["weight_update_ratio"].split("-")
    manual_weight_idx = backward_config["manual_weight_idx"].split("-")
    budget = 0
    for i in range(len(manual_weight_idx)):
        budget += float(weight_update_ratio[i])*num_params[int(manual_weight_idx[i])]
    
    print("Total updatable num param: ", budget)


def compute_update_budget(num_conv_params, ratio):
    return int(num_conv_params * ratio)


def compute_Conv2d_flops(module):
    assert isinstance(module, nn.Conv2d)
    assert len(module.input_shape) == 4 and len(module.input_shape) == len(
        module.output_shape
    )

    # counting the FLOPS for one input
    in_c = module.input_shape[1]
    k_h, k_w = module.kernel_size
    out_c, out_h, out_w = module.output_shape[1:]
    groups = module.groups

    filters_per_channel = out_c // groups
    conv_per_position_flops = k_h * k_w * in_c * filters_per_channel
    active_elements_count = out_h * out_w

    total_conv_flops = conv_per_position_flops * active_elements_count

    bias_flops = 0
    if module.bias is not None:
        bias_flops = out_c * active_elements_count

    total_flops = total_conv_flops + bias_flops
    return total_flops


@torch.no_grad()
def find_module_by_name(model, name):
    module = model
    splitted_name = name.split(".")
    for idx, sub in enumerate(splitted_name):
        if idx < len(splitted_name):
            module = getattr(module, sub)

    return module


# Set module gradients to 0 given neurons to freeze
@torch.no_grad()
def zero_gradients(model, name, mask):
    module = find_module_by_name(model, name) # only find convolution layer because name is the name of convolution layer, which is taken from grad_mask (also is created for convolution layers)
    module.weight.grad[mask] = 0.0
    # if getattr(module, "bias", None) is not None:
    #     module.bias.grad[mask] = 0.0
    # TODO : implement bias freezing depending on depth for SU update : here it's not important as mbv2 has no bias
    all_conv_layers = get_all_conv_ops(model) # Get all convolution layers in the model
    for index in range(len(all_conv_layers) - config.backward_config["n_bias_update"]): # Travel through all layers having bias updated
        if getattr(all_conv_layers[index], "bias", None) is not None:
            all_conv_layers[index].bias.grad[:] = 0.0 # Freeze all biases in this layer (or can use 3 lines of code below instead)
            # for name, param in all_conv_layers[index].named_parameters():
            #     if name == "bias":
            #         param.grad.zero_()


@torch.no_grad()
def zero_all_gradients(model):
    for module in model.modules():
        if getattr(module, "weight", None) is not None:
            if not isinstance(module, nn.Linear):
                module.weight.grad = torch.zeros_like(module.weight.grad)
                if getattr(module, "bias", None) is not None:
                    module.bias.grad = torch.zeros_like(module.bias.grad)


def reshape(output):
    reshaped_output = output.view(
        (output.shape[0], output.shape[1], -1)
        if len(output.shape) > 2
        else (output.shape[0], output.shape[1])
    ).mean(dim=2)
    return reshaped_output


def log_masks(model, hooks, grad_mask, total_neurons, total_conv_flops):
    frozen_neurons = 0
    total_saved_flops = 0

    per_layer_frozen_neurons = {}
    per_layer_saved_flops = {}

    for k in grad_mask:
        frozen_neurons += grad_mask[k].shape[0]

        module = find_module_by_name(model, k)

        layer_flops = hooks[k].flops
        saved_layer_flops = grad_mask[k].shape[0] / module.weight.shape[0] * layer_flops
        total_saved_flops += saved_layer_flops
        # Log the percentage of frozen neurons per layer
        per_layer_frozen_neurons[f"{k}"] = (
            grad_mask[k].shape[0] / module.weight.shape[0] * 100
        )
        per_layer_saved_flops[f"{k}"] = saved_layer_flops

    # Log the total percentage of frozen neurons
    return (
        {
            "total": frozen_neurons / total_neurons * 100,
            "layer": per_layer_frozen_neurons,
        },
        {
            "total": total_saved_flops / total_conv_flops * 100,
            "layer": per_layer_saved_flops,
        },
    )


# Call this function to access to a network's number of convolutional parameters
if __name__ == "__main__":
    # load_transfer_config("transfer.yaml")#load_config_from_file("configs/transfer.yaml")
    net_name = "proxyless-w0.3" # Fill in the name of the model needed to be measured
    schemes = ["proxyless-w0.3_scheme_1", "proxyless-w0.3_scheme_2", "proxyless-w0.3_scheme_3", "proxyless-w0.3_scheme_4", "proxyless-w0.3_scheme_5"]
    if "mcunet" in net_name or net_name == "proxyless-w0.3":
        model, _, _, _ = get_model(net_name)
    else:
        model, _ = get_model(net_name)
    for scheme in schemes:
        count_updateable_param(model, scheme)
    # count_net_num_conv_params(model)