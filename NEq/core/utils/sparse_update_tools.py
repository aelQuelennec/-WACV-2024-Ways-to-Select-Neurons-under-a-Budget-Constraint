import torch
from core.utils.config import config


def _is_depthwise_conv(conv):
    return conv.groups == conv.in_channels == conv.out_channels


def _is_pw1(conv):  # for mbnets
    return conv.out_channels > conv.in_channels and conv.kernel_size == (
        1,
        1,
    )


def get_all_conv_ops(model):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    # sanity check, do not include the final fc layer
    return [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]


def parsed_backward_config(backward_config, model):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    n_conv = len(get_all_conv_ops(model))
    # parse config (if None, update all)
    if backward_config["n_bias_update"] == "all":
        backward_config["n_bias_update"] = n_conv
    else:
        assert isinstance(backward_config["n_bias_update"], int), backward_config[
            "n_bias_update"
        ]

    # get the layer update index
    backward_config["manual_weight_idx"] = [
        int(p) for p in str(backward_config["manual_weight_idx"]).split("-")
    ]

    # sanity check: the weight update layers all update bias
    for idx in backward_config["manual_weight_idx"]:
        assert idx in [
            n_conv - 1 - i_w for i_w in range(backward_config["n_bias_update"])
        ]

    n_weight_update = len(backward_config["manual_weight_idx"])
    if backward_config["weight_update_ratio"] is None:
        backward_config["weight_update_ratio"] = [None] * n_weight_update
    elif isinstance(
        backward_config["weight_update_ratio"], (int, float)
    ):  # single number
        assert backward_config["weight_update_ratio"] <= 1
        backward_config["weight_update_ratio"] = [
            backward_config["weight_update_ratio"]
        ] * n_weight_update
    else:  # list
        backward_config["weight_update_ratio"] = [
            float(p) for p in backward_config["weight_update_ratio"].split("-")
        ]
        assert len(backward_config["weight_update_ratio"]) == n_weight_update
    # if we update weights, let's also update bias
    assert backward_config["n_bias_update"] >= n_weight_update
    return backward_config


def manually_initialize_grad_mask(
    hooks, grad_mask, model, backward_config, log_num_saved_params
):
    def _get_conv_w_norm(_conv):
        _o, _i, _h, _w = _conv.weight.shape
        w_norm = torch.norm(_conv.weight.data.reshape(_o, -1), dim=1)
        assert w_norm.numel() == _conv.out_channels
        return w_norm

    # assume sorted
    assert backward_config["manual_weight_idx"] == sorted(
        backward_config["manual_weight_idx"]
    ), backward_config["manual_weight_idx"]

    # select the channels to be update
    conv_ops = get_all_conv_ops(model)
    ratio_ptr = 0
    num_saved_params = 0
    num_saved_neurons = 0
    for (i_conv, conv), k in zip(enumerate(conv_ops), hooks):  # from input to output
        if (
            i_conv in backward_config["manual_weight_idx"]
        ):  # the weight is updated for this layer
            keep_ratio = backward_config["weight_update_ratio"][ratio_ptr]
            ratio_ptr += 1
            n_freeze = int(conv.out_channels * (1 - keep_ratio))
            num_saved_neurons += conv.out_channels - n_freeze
            w_norm = _get_conv_w_norm(conv)
            grad_mask[k] = torch.argsort(w_norm)[:n_freeze]
            if _is_depthwise_conv(conv):  # depthwise
                weight_shape = conv.weight.shape  # o, 1, k, k
                this_num_weight = conv.in_channels * weight_shape[2] * weight_shape[3]
            else:
                weight_shape = conv.weight.shape  # o, i, k, k
                if conv.groups == 1:  # normal conv
                    this_num_weight = (
                        weight_shape[0]
                        * conv.in_channels
                        * weight_shape[2]
                        * weight_shape[3]
                    )
                else:  # group conv (lite residual)
                    this_num_weight = conv.weight.data.numel()
            num_saved_params += int(this_num_weight * keep_ratio)
        else:  # this layer is completely frozen
            grad_mask[k] = torch.tensor(range(0, conv.out_channels))

    log_num_saved_params["Number of saved parameters"] = num_saved_params
    log_num_saved_params["Parameters delta with Budget"] = (
        config.NEq_config.glob_num_params - num_saved_params
    )
    log_num_saved_params["Number of saved neurons"] = num_saved_neurons
