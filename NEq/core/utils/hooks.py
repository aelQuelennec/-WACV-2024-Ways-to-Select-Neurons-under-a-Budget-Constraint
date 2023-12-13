from copy import deepcopy
import torch
from torch import nn

from core.utils.config import config
from core.utils.neuron_selections_methods import select_mask_method
from compute_stats import plots


def record_in_out_shape(m_, x, y):
    x = x[0]
    m_.input_shape = list(x.shape)
    m_.output_shape = list(y.shape)


def add_activation_shape_hook(m_):
    m_.register_forward_hook(record_in_out_shape)


def attach_hooks(model, hooks):
    for n, m in model.named_modules():
        # if n in arch_config["targets"]:
        # if isinstance(m, (nn.Conv2d, nn.BatchNorm2d)):
        if isinstance(m, (nn.Conv2d)):
            hooks[n] = Hook(n, m, config.NEq_config.velocity_mu)


def activate_hooks(hooks, active):
    for h in hooks:
        hooks[h].activate(active)


class Hook:
    def __init__(self, name, module, momentum=0) -> None:
        self.name = name
        self.module = module
        self.samples_activation = []
        self.previous_activations = None
        self.activation_deltas = 0
        self.total_samples = 0

        self.momentum = momentum
        self.delta_buffer = 0
        self.velocity = 0

        self.active = True

        self.single_neuron_num_params = (
            self.module.in_channels
            * self.module.weight.shape[2]
            * self.module.weight.shape[3]
        )
        if self.module.bias is not None:
            self.single_neuron_num_params += 1

        self.flops = 0

        self.hook = module.register_forward_hook(self.hook_fn)


    def hook_fn(
        self, module: torch.nn.Module, input: torch.Tensor, output: torch.Tensor
    ) -> None:
        if not self.active:
            return

        reshaped_output = output.view(
            (output.shape[0], output.shape[1], -1)
            if len(output.shape) > 2
            else (output.shape[0], output.shape[1])
        )

        if self.previous_activations is None:
            self.samples_activation.append(reshaped_output)
        else:
            previous = self.previous_activations[
                self.total_samples : output.shape[0] + self.total_samples
            ].float()
            delta = 1 - cosine_similarity(
                reshaped_output.float(),
                previous,
                dim=0 if len(output.shape) <= 2 else 2,
            )

            if len(output.shape) > 2:
                delta = torch.sum(delta, dim=0)

            self.activation_deltas += delta

            self.previous_activations[
                self.total_samples : output.shape[0] + self.total_samples
            ] = reshaped_output
            self.total_samples += output.shape[0]


    def get_samples_activation(self):
        return torch.cat(self.samples_activation)


    def get_reduced_activation_delta(self):
        return self.activation_deltas / self.total_samples


    def get_delta_of_delta(self):
        reduced_activation_delta = self.get_reduced_activation_delta()
        delta_of_delta = self.delta_buffer - reduced_activation_delta

        return delta_of_delta


    def get_velocity(self):
        self.velocity += self.get_delta_of_delta()

        return self.velocity


    def update_delta_buffer(self):
        self.delta_buffer = self.get_reduced_activation_delta()


    def update_velocity(self):
        self.velocity *= self.momentum
        self.velocity -= self.get_delta_of_delta()


    def reset(self, previous_activations=None):
        self.samples_activation = []
        self.activation_deltas = 0
        self.total_samples = 0
        if previous_activations is not None:
            self.previous_activations = previous_activations


    def close(self) -> None:
        self.hook.remove()


    def activate(self, active):
        self.active = active


def cosine_similarity(x1, x2, dim, eps=1e-8):
    x1_squared_norm = torch.pow(x1, 2).sum(dim=dim, keepdim=True)
    x2_squared_norm = torch.pow(x2, 2).sum(dim=dim, keepdim=True)

    # x1_squared_norm.clamp_min_(eps)
    # x2_squared_norm.clamp_min_(eps)

    x1_norm = x1_squared_norm.sqrt_()
    x2_norm = x2_squared_norm.sqrt_()

    x1_normalized = x1.div(x1_norm).nan_to_num(nan=0, posinf=0, neginf=0)
    x2_normalized = x2.div(x2_norm).nan_to_num(nan=0, posinf=0, neginf=0)

    mask_1 = (torch.abs(x1_normalized).sum(dim=dim) <= eps) * (
        torch.abs(x2_normalized).sum(dim=dim) <= eps
    )
    mask_2 = (torch.abs(x1_normalized).sum(dim=dim) > eps) * (
        torch.abs(x2_normalized).sum(dim=dim) > eps
    )

    cos_sim_value = torch.sum(x1_normalized * x2_normalized, dim=dim)

    return mask_2 * cos_sim_value + mask_1


def get_global_gradient_mask(log_num_saved_params, hooks, grad_mask, epoch):
    velocity_list = []
    hooks_num_params_list = []

    for k in hooks:
        _ = deepcopy(hooks[k].get_reduced_activation_delta().detach().clone())
        _ = deepcopy(hooks[k].get_delta_of_delta().detach().clone())
        velocity = deepcopy(hooks[k].get_velocity().detach().clone())
        velocity_list.append(velocity / hooks[k].single_neuron_num_params)

        hooks_num_params_list.append(
            torch.Tensor([hooks[k].single_neuron_num_params] * len(velocity))
        )

        hooks[k].update_velocity()
        hooks[k].update_delta_buffer()

        hooks[k].reset()

    select_mask_method(
        hooks,
        grad_mask,
        velocity_list,
        hooks_num_params_list,
        log_num_saved_params,
        epoch,
    )
