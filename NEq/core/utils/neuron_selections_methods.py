import numpy as np
import torch

from core.utils.config import config


# Computes the velocity threshold and corresponding mask such that we update less parameters than the SU equivalent
def _compute_velocity_budget_mask(
    hooks, grad_mask, velocity_list, hooks_num_params_list, log_num_saved_params
):
    sorted_velocities = torch.sort(torch.abs(torch.cat(velocity_list)), descending=True)
    hooks_num_params = torch.cat(hooks_num_params_list)
    current_sum = 0
    new_sum = 0
    sum_index = 0
    new_index = 0
    while new_sum <= config.NEq_config.glob_num_params:
        current_sum = new_sum
        sum_index = new_index
        new_sum += hooks_num_params[sorted_velocities.indices[sum_index]].item()
        new_index += 1
    best_neurons_threshold = sorted_velocities.values[sum_index]

    log_num_saved_params["Number of saved parameters"] = current_sum
    log_num_saved_params["Parameters delta with Budget"] = (
        config.NEq_config.glob_num_params - current_sum
    )
    log_num_saved_params["Number of saved neurons"] = sum_index

    for k, velocity in zip(hooks, velocity_list):
        mask = torch.where(torch.abs(velocity) <= best_neurons_threshold)[0]
        grad_mask[k] = mask


# Computes a random mask such that we update less parameters than the SU equivalent
def compute_random_budget_mask(
    hooks, grad_mask, hooks_num_params_list, log_num_saved_params
):
    slices_list = []
    slice_index = 0
    for elt in hooks_num_params_list:
        slices_list.append((slice_index, slice_index + len(elt)))
        slice_index += len(elt)
    hooks_num_params = torch.cat(hooks_num_params_list)
    num_neurons = len(hooks_num_params)
    indices = torch.Tensor(range(num_neurons))
    hooks_num_params_with_indices = torch.stack((hooks_num_params, indices))
    permutation = torch.randperm(num_neurons)
    shuffled_hooks = hooks_num_params_with_indices[:, permutation]
    selected_neurons = torch.Tensor([0] * num_neurons)
    current_sum = 0
    new_sum = 0
    sum_index = 0
    new_index = 0
    while new_sum <= config.NEq_config.glob_num_params:
        selected_neurons[sum_index] = 1
        current_sum = new_sum
        sum_index = new_index
        new_sum += shuffled_hooks[0][sum_index].item()
        new_index += 1

    sort_indices = shuffled_hooks[1].sort().indices
    sorted_neurons = selected_neurons[sort_indices]

    for k, slice_indices in zip(hooks, slices_list):
        mask = torch.where(sorted_neurons[slice_indices[0] : slice_indices[1]] < 1)[0]
        grad_mask[k] = mask

    log_num_saved_params["Number of saved parameters"] = current_sum
    log_num_saved_params["Parameters delta with Budget"] = (
        config.NEq_config.glob_num_params - current_sum
    )
    log_num_saved_params["Number of saved neurons"] = sum_index


# Outputs empty mask to update all the neurons
def compute_full_update(hooks, grad_mask, hooks_num_params_list, log_num_saved_params):
    num_saved_neurons = 0
    num_saved_params = 0
    i = 0
    for k in hooks:
        grad_mask[k] = torch.LongTensor([])
        num_saved_neurons += len(hooks_num_params_list[i])
        num_saved_params += torch.sum(hooks_num_params_list[i])
        i += 1

    log_num_saved_params["Number of saved neurons"] = num_saved_neurons
    log_num_saved_params["Number of saved parameters"] = num_saved_params


def select_mask_method(
    hooks, grad_mask, velocity_list, hooks_num_params_list, log_num_saved_params, epoch
):
    if config.NEq_config.neuron_selection == "velocity":
        _compute_velocity_budget_mask(
            hooks,
            grad_mask,
            velocity_list,
            hooks_num_params_list,
            log_num_saved_params,
        )
    elif config.NEq_config.neuron_selection == "random":
        compute_random_budget_mask(
            hooks, grad_mask, hooks_num_params_list, log_num_saved_params
        )
    elif config.NEq_config.neuron_selection == "full":
        compute_full_update(
            hooks, grad_mask, hooks_num_params_list, log_num_saved_params
        )