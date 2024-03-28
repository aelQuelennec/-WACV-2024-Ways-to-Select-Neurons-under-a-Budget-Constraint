import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import sys
import torch
import argparse
import wandb
from torch import nn
import torch.utils.data.distributed

from classification.models import get_model
from classification.utils import get_optimizer
from core.builder.lr_scheduler import build_lr_scheduler
from core.dataset import build_dataset
from core.trainer.cls_trainer import ClassificationTrainer
from core.utils import dist
from core.utils.config import (
    config,
    load_transfer_config,
    update_config_from_wandb,
)
from core.utils.hooks import (
    attach_hooks,
    activate_hooks,
    add_activation_shape_hook,
)
from core.utils.logging import logger
from general_utils import (
    set_seed,
    change_classifier_head,
    find_module_by_name,
    compute_Conv2d_flops,
)


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--config-file",
        type=str,
        default="transfer",
        help="Which config file to call for training",
    )

    parsed = parser.parse_args()

    return f"{parsed.config_file}.yaml"


def build_config(config_file_path):
    load_transfer_config(config_file_path)

    # Init wandb
    if dist.rank() <= 0:
        print("Initialize wandb run")
        wandb.init(project=config.project_name, config=config)
        os.makedirs(os.path.join("./scratch", "checkpoints", wandb.run.id))

    if config.wandb_sweep:
        update_config_from_wandb(wandb.config)
        wandb.config.update(config, allow_val_change=True)


def main():
    # Set reproducibility
    set_seed(config.manual_seed)

    # Launch multi-core computation
    dist.init()
    torch.cuda.set_device(dist.local_rank())

    assert config.run_dir is not None
    os.makedirs(config.run_dir, exist_ok=True)
    logger.init()  # dump exp config
    logger.info(" ".join([sys.executable] + sys.argv))
    logger.info(f'Experiment started: "{config.run_dir}".')

    # Loading model
    print(f"Initialize model {config.net_config.net_name}")
    if "mcunet" in config.net_config.net_name or config.net_config.net_name == "proxyless-w0.3":
        model, config.data_provider.image_size, description, total_neurons = get_model(config.net_config.net_name)

        print("Model: ", config.net_config.net_name)
        print("image_size: ", config.data_provider.image_size)
        print("Description: ", description)
        print("Total neuron: ", total_neurons)
    else:
        model, total_neurons = get_model(config.net_config.net_name)

        print("Model: ", config.net_config.net_name)
        print("Total neuron: ", total_neurons)

    dataset = build_dataset()
    data_loader = dict()
    for split in dataset:
        sampler = torch.utils.data.DistributedSampler(
            dataset[split],
            num_replicas=dist.size(),
            rank=dist.rank(),
            seed=config.manual_seed,
            shuffle=(split == "train"),
        )
        data_loader[split] = torch.utils.data.DataLoader(
            dataset[split],
            batch_size=config.data_provider.base_batch_size,
            sampler=sampler,
            num_workers=config.data_provider.n_worker,
            pin_memory=True,
            drop_last=(split == "train"),
        )
    
    if "mbv2" in config.net_config.net_name:
        classifier = model.classifier[1]
    elif "mcunet" in config.net_config.net_name or config.net_config.net_name == "proxyless-w0.3":
        classifier = model.classifier # Based on information from NEq/classification/models/__init__/get_model/cfg
    else:
        classifier = model.fc
    # Change classifier head in case of fine-tuning
    if config.net_config.fine_tuning:
        change_classifier_head(classifier)

    # Registering input and output shapes for each module
    model.apply(add_activation_shape_hook)

    # setting the model to work on GPUs
    model.cuda()
    if dist.size() > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[dist.local_rank()]
        )

    # Build optimizer and scheduler
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = get_optimizer(model)
    lr_scheduler = build_lr_scheduler(optimizer, len(data_loader["train"]))

    # Init dictionaries
    hooks = {}
    previous_activations = {}
    grad_mask = {}

    trainer = ClassificationTrainer(
        model,
        data_loader,
        criterion,
        optimizer,
        lr_scheduler,
        hooks,
        grad_mask,
        classifier,
    )

    # Resume training in case it was stopped
    if config.resume:
        trainer.resume()

    # Attach the hooks used to gather the PSP value
    attach_hooks(trainer.model, trainer.hooks)

    # First run on validation to get the PSP for epoch -1
    if wandb.config.scheme != "scheme_baseline":
        activate_hooks(trainer.hooks, wandb.config.scheme != "scheme_baseline")  # When baseline is used, I dont need to use velocity and neuron selection method => Deactivate the hook
        _ = trainer.validate("val_velocity")

    total_conv_flops = 0
    if (wandb.config.scheme != "scheme_baseline"):#if (not use_baseline):
        # Save the activations into the dict + compute flops per conv layer
        for k in trainer.hooks:
            previous_activations[k] = trainer.hooks[k].get_samples_activation()
            trainer.hooks[k].reset(previous_activations[k])
            module = find_module_by_name(model, k)
            layer_flops = compute_Conv2d_flops(module)
            trainer.hooks[k].flops = layer_flops
            total_conv_flops += layer_flops

    # Training the model
    val_info_dict = trainer.run_training(total_neurons, total_conv_flops, wandb.config.scheme)

    if dist.rank() <= 0:
        wandb.run.finish()

    return val_info_dict


if __name__ == "__main__":
    config_file_path = get_parser()
    build_config(config_file_path)
    main()