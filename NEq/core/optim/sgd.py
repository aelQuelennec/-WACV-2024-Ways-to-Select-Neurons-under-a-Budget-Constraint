import torch


class MaskedSGD(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        names,
        lr,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        masks=None,
    ):
        if masks is None:
            masks = {}
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(
            names=names,
            lr=lr,
            masks=masks,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(MaskedSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(MaskedSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]
            lr = group["lr"]
            masks = group["masks"]

            for p, n in zip(group["params"], group["names"]):
                if p.grad is not None:
                    params_with_grad.append((p, n))
                    d_p_list.append(p.grad)

                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state["momentum_buffer"])

            sgd(
                params_with_grad,
                d_p_list,
                momentum_buffer_list,
                weight_decay=weight_decay,
                momentum=momentum,
                lr=lr,
                masks=masks,
                dampening=dampening,
                nesterov=nesterov,
            )

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                p, n = p

                state = self.state[p]
                state["momentum_buffer"] = momentum_buffer

        return loss


def sgd(
    params,
    d_p_list,
    momentum_buffer_list,
    *,
    weight_decay,
    momentum,
    lr,
    masks,
    dampening,
    nesterov
):
    for i, param in enumerate(params):
        param, name = param
        root_name = name.replace(".weight", "").replace(".bias", "")

        d_p = d_p_list[i]

        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)

        if root_name in masks:
            d_p[masks[root_name]] = 0.0

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

            if nesterov:
                d_p = d_p.add(buf, alpha=momentum)
            else:
                d_p = buf

        if root_name in masks:
            d_p[masks[root_name]] = 0.0

        param.add_(d_p, alpha=-lr)
