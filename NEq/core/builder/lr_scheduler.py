import torch
from torch.optim.lr_scheduler import MultiStepLR
from typing import List
import math
from ..utils import dist
from ..utils.config import config

__all__ = ["build_lr_scheduler", "CosineLRwithWarmup"]


def build_lr_scheduler(optimizer, batch_per_epoch):
    if config.run_config.lr_schedule_name == "cosine":
        lr_scheduler = CosineLRwithWarmup(
            optimizer,
            config.run_config.warmup_epochs * batch_per_epoch,
            config.run_config.warmup_lr * dist.size(),
            config.run_config.n_epochs * batch_per_epoch,
            final_lr=config.run_config.get("final_lr", 0) * dist.size(),
        )
    elif config.run_config.lr_schedule_name == "gamma_step":
        lr_scheduler = StepLRwithWarmup(
            optimizer,
            config.run_config.warmup_epochs * batch_per_epoch,
            config.run_config.warmup_lr * dist.size(),
            config.run_config.get("lr_step_size", 30) * batch_per_epoch,
            config.run_config.get("lr_step_gamma", 0.1),
        )
    elif config.run_config.lr_schedule_name == "multi_step":
        lr_scheduler = MultiStepLR(
            optimizer, milestones=[100 * batch_per_epoch, 150 * batch_per_epoch]
        )
    else:
        raise NotImplementedError(config.run_config.lr_schedule_name)
    return lr_scheduler


class CosineLRwithWarmup(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        warmup_lr: float,
        decay_steps: int,
        final_lr: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_steps = warmup_steps
        self.warmup_lr = warmup_lr
        self.decay_steps = decay_steps
        self.final_lr = final_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if self.last_epoch < self.warmup_steps:
            return [
                (base_lr - self.warmup_lr) * self.last_epoch / self.warmup_steps
                + self.warmup_lr
                for base_lr in self.base_lrs
            ]
        else:
            current_steps = self.last_epoch - self.warmup_steps
            return [
                0.5
                * (base_lr - self.final_lr)
                * (1 + math.cos(math.pi * current_steps / self.decay_steps))
                + self.final_lr
                for base_lr in self.base_lrs
            ]


class StepLRwithWarmup(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        warmup_lr: float,
        step_size: int,
        gamma: float = 0.1,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_steps = warmup_steps
        self.warmup_lr = warmup_lr
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if self.last_epoch < self.warmup_steps:
            return [
                (base_lr - self.warmup_lr) * self.last_epoch / self.warmup_steps
                + self.warmup_lr
                for base_lr in self.base_lrs
            ]
        else:
            current_steps = self.last_epoch - self.warmup_steps
            n_decay = current_steps // self.step_size
            return [base_lr * (self.gamma**n_decay) for base_lr in self.base_lrs]
