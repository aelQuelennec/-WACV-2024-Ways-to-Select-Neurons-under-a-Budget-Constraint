from copy import deepcopy
import os
import torch
import torch.nn as nn
import wandb
from collections import defaultdict

from ..utils import dist
from ..utils.config import config
from ..utils.logging import logger
from ..utils.sparse_update_tools import (
    parsed_backward_config,
    manually_initialize_grad_mask,
)
from ..utils.neuron_selections_methods import (
    compute_random_budget_mask,
    compute_full_update,
)
from ..utils.hooks import (
    activate_hooks,
    get_global_gradient_mask,
)
from general_utils import (
    log_masks,
)

__all__ = ["BaseTrainer"]


class BaseTrainer(object):
    def __init__(
        self,
        model: nn.Module,
        data_loader,
        criterion,
        optimizer,
        lr_scheduler,
        hooks,
        grad_mask,
        classifier,
    ):
        self.model = model
        self.classifier = classifier
        self.data_loader = data_loader
        self.criterion = criterion

        self.best_test = 0.0
        self.test_top1_at_best_val = 0.0  # Log top-1 test accuracy of the model when its top-1 validation accuracy reachs the highest value
        self.best_val = 0.0
        self.start_epoch = 0

        # optimization-related
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        # NEq related
        self.hooks = hooks
        self.grad_mask = grad_mask

    @property
    def checkpoint_path(self):
        return os.path.join(config.run_dir, "checkpoint")

    def save(self, epoch=0, is_best=False):
        if dist.rank() == 0:
            checkpoint = {
                "state_dict": self.model.module.state_dict()
                if isinstance(self.model, nn.parallel.DistributedDataParallel)
                else self.model.state_dict(),
                "epoch": epoch,
                "best_val": self.best_val,
                "best_test": self.best_test,
                "test_top1_at_best_val": self.test_top1_at_best_val,
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
            }

            os.makedirs(self.checkpoint_path, exist_ok=True)
            torch.save(checkpoint, os.path.join(self.checkpoint_path, "ckpt.pth"))

            if is_best:
                torch.save(
                    checkpoint, os.path.join(self.checkpoint_path, "ckpt.best.pth")
                )

    def resume(self):
        model_fname = os.path.join(self.checkpoint_path, "ckpt.pth")
        if os.path.exists(model_fname):
            checkpoint = torch.load(model_fname, map_location="cpu")

            # load checkpoint
            if hasattr(self.model, "module"):
                self.model.module.load_state_dict(checkpoint["state_dict"])
            else:
                self.model.load_state_dict(checkpoint["state_dict"])
            if "epoch" in checkpoint:
                self.start_epoch = checkpoint["epoch"] + 1
                logger.info("loaded epoch: %d" % checkpoint["epoch"])
            else:
                logger.info("!!! epoch not found in checkpoint")
            if "best_test" in checkpoint:
                self.best_test = checkpoint["best_test"]
                logger.info("loaded best_test: %f" % checkpoint["best_test"])
            else:
                logger.info("!!! best_test not found in checkpoint")
            if "best_val" in checkpoint:
                self.best_val = checkpoint["best_val"]
                logger.info("loaded best_val: %f" % checkpoint["best_val"])
            else:
                logger.info("!!! best_val not found in checkpoint")
            if "test_top1_at_best_val" in checkpoint:
                self.test_top1_at_best_val = checkpoint["test_top1_at_best_val"]
                logger.info(
                    "loaded test_top1_at_best_val: %f"
                    % checkpoint["test_top1_at_best_val"]
                )
            else:
                logger.info("!!! test_top1_at_best_val not found in checkpoint")
            if "optimizer" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                logger.info("loaded optimizer")
            else:
                logger.info("!!! optimizer not found in checkpoint")
            if "lr_scheduler" in checkpoint:
                self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
                logger.info("loaded lr_scheduler")
            else:
                logger.info("!!! lr_scheduler not found in checkpoint")
        else:
            logger.info("Skipping resume... checkpoint not found")

    def validate(self):
        raise NotImplementedError

    def train_one_epoch(self, epoch):
        raise NotImplementedError

    def run_training(self, total_neurons, total_conv_flops, config_scheme):
        test_info_dict = None
        val_info_dict = None

        for epoch in range(
            self.start_epoch,
            config.run_config.n_epochs + config.run_config.warmup_epochs,
        ):
            if epoch == 0:
                log_num_saved_params = {}

                # Selecting neurons to update for first epoch from SU config
                if config.NEq_config.initialization == "SU":
                    config.backward_config = parsed_backward_config(
                        config.backward_config, self.model
                    )
                    manually_initialize_grad_mask(
                        self.hooks,
                        self.grad_mask,
                        self.model,
                        config.backward_config,
                        log_num_saved_params,
                    )

                # Randomly selecting neurons to update for first epoch
                elif "random" in config.NEq_config.initialization:
                    hooks_num_params_list = []
                    for k in self.hooks:
                        hooks_num_params_list.append(
                            torch.Tensor(
                                [self.hooks[k].single_neuron_num_params]
                                * self.hooks[k].module.out_channels
                            )
                        )
                    compute_random_budget_mask(
                        self.hooks,
                        self.grad_mask,
                        hooks_num_params_list,
                        log_num_saved_params,
                    )

                # Updating all the neurons for first epoch
                elif "full" in config.NEq_config.initialization:
                    hooks_num_params_list = []
                    for k in self.hooks:
                        hooks_num_params_list.append(
                            torch.Tensor(
                                [self.hooks[k].single_neuron_num_params]
                                * self.hooks[k].module.out_channels
                            )
                        )
                    compute_full_update(
                        self.hooks,
                        self.grad_mask,
                        hooks_num_params_list,
                        log_num_saved_params,
                    )

            # Log the amount of frozen neurons
            use_baseline = config_scheme == "scheme_baseline"
            if not use_baseline:
                frozen_neurons, saved_flops = log_masks(
                    self.model,
                    self.hooks,
                    self.grad_mask,
                    total_neurons,
                    total_conv_flops,
                )

            # Train step
            activate_hooks(self.hooks, False)
            train_info_dict = self.train_one_epoch(epoch)
            logger.info(f"epoch {epoch}: f{train_info_dict}")

            # Validation step to compute neurons velocities
            if config.NEq_config.neuron_selection == "velocity":
                activate_hooks(self.hooks, True)
                self.validate("val_velocity")

            # Validate after each epoch
            if config.data_provider.use_validation:
                activate_hooks(self.hooks, False)
                val_info_dict = self.validate("val")
                is_best_val = val_info_dict["val/top1"] > self.best_val
                self.best_val = max(val_info_dict["val/top1"], self.best_val)
                if is_best_val:
                    logger.info(
                        " * New best val acc (epoch {}): {:.2f}".format(
                            epoch, self.best_val
                        )
                    )
                val_info_dict["val/best"] = self.best_val
                logger.info(f"epoch {epoch}: {val_info_dict}")

            # Testing step to observe accuracy evolution (after each test_per_epochs or at the last epoch)
            if (
                (epoch + 1) % config.run_config.test_per_epochs == 0
                or epoch
                == config.run_config.n_epochs + config.run_config.warmup_epochs - 1
            ):
                activate_hooks(self.hooks, False)
                test_info_dict = self.validate("test")
                is_best_test = test_info_dict["test/top1"] > self.best_test
                self.best_test = max(test_info_dict["test/top1"], self.best_test)
                # Testing step to observe accuracy when validation top 1 acc is best (only in case validation set is used):
                if config.data_provider.use_validation:
                    if is_best_val:
                        self.test_top1_at_best_val = test_info_dict["test/top1"]
                        logger.info(
                            " * Valid/best at epoch {}, with Test/top1 is {:.2f}".format(
                                epoch, test_info_dict["test/top1"]
                            )
                        )
                    test_info_dict[
                        "test/top1_at_valid_best"
                    ] = self.test_top1_at_best_val

                if is_best_test:
                    logger.info(
                        " * New best test acc (epoch {}): {:.2f}".format(
                            epoch, self.best_test
                        )
                    )
                test_info_dict["test/best"] = self.best_test
                logger.info(f"epoch {epoch}: {test_info_dict}")
                #############################################

            # save model when validation acc reach its highest value, in case validation set is not used -> save model when best test
            if config.data_provider.use_validation:
                self.save(
                    epoch=epoch,
                    is_best=is_best_val,
                )
            else:
                self.save(
                    epoch=epoch,
                    is_best=is_best_test,
                )

            # Logs
            if dist.rank() <= 0:
                if use_baseline:
                    saved_flops = None
                wandb.log(
                    {
                        "Perc of frozen conv neurons": frozen_neurons,
                        "FLOPS stats": saved_flops,
                        "train": train_info_dict,
                        "valid": val_info_dict,
                        "test": test_info_dict,
                        "epochs": epoch,
                        "lr": self.optimizer.param_groups[0]["lr"],
                        "Saved parameters": log_num_saved_params,
                    }
                )

            # Not reseting grad mask and log in case of static selection
            if (
                not config.NEq_config.neuron_selection == "SU"
                or not config.NEq_config.neuron_selection == "full"
            ):
                self.grad_mask = {}
                log_num_saved_params = {}
                get_global_gradient_mask(
                    log_num_saved_params, self.hooks, self.grad_mask, epoch
                )

            elif (
                epoch == 0 and not config.NEq_config.initialization == "SU"
            ):  # in case SU selection but not initialization, then init SU mask.
                self.grad_mask = {}
                log_num_saved_params = {}
                config.backward_config = parsed_backward_config(
                    config.backward_config, self.model
                )
                manually_initialize_grad_mask(
                    self.hooks,
                    self.grad_mask,
                    self.model,
                    config.backward_config,
                    log_num_saved_params,
                )
