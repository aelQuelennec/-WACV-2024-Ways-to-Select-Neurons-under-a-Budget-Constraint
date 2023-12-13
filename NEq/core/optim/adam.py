import math

import torch
from torch.optim import Optimizer


class MaskedAdam(Optimizer):
    def __init__(
        self,
        params,
        names,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
        masks=None,
    ):
        if masks is None:
            masks = {}
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            names=names,
            masks=masks,
        )
        super(MaskedAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(MaskedAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group["betas"]
            masks = group["masks"]

            for p, n in zip(group["params"], group["names"]):
                if p.grad is not None:
                    params_with_grad.append((p, n))
                    if p.grad.is_sparse:
                        raise RuntimeError(
                            "Adam does not support sparse gradients, please consider SparseAdam instead"
                        )
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state["step"] = 0
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )
                        # Exponential moving average of squared gradient values
                        state["exp_avg_sq"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )
                        if group["amsgrad"]:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state["max_exp_avg_sq"] = torch.zeros_like(
                                p, memory_format=torch.preserve_format
                            )

                    exp_avgs.append(state["exp_avg"])
                    exp_avg_sqs.append(state["exp_avg_sq"])

                    if group["amsgrad"]:
                        max_exp_avg_sqs.append(state["max_exp_avg_sq"])

                    # update the steps for each param group update
                    state["step"] += 1
                    # record the step after step update
                    state_steps.append(state["step"])

            adam(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                amsgrad=group["amsgrad"],
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                masks=masks,
            )

        return loss


def adam(
    params,
    grads,
    exp_avgs,
    exp_avg_sqs,
    max_exp_avg_sqs,
    state_steps,
    *,
    amsgrad,
    beta1,
    beta2,
    lr,
    weight_decay,
    eps,
    masks
):
    for i, param in enumerate(params):
        param, name = param
        root_name = name.replace(".weight", "").replace(".bias", "")

        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = state_steps[i]

        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        if root_name in masks:
            grad[masks[root_name]] = 0.0

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)
        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
            # Use the max. for normalizing running avg. of gradient
            denom = (max_exp_avg_sqs[i].sqrt() / math.sqrt(bias_correction2)).add_(eps)
        else:
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

        step_size = lr / bias_correction1

        if root_name in masks:
            exp_avg[masks[root_name]] = 0.0

        param.addcdiv_(exp_avg, denom, value=-step_size)
