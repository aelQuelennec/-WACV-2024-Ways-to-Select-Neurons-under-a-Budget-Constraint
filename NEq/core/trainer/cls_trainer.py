from copy import deepcopy
import os
import pickle
from tqdm import tqdm
import torch

from .base_trainer import BaseTrainer
from ..utils.basic import DistributedMetric, accuracy
from ..utils.config import config
from ..utils import dist
from general_utils import zero_gradients


class ClassificationTrainer(BaseTrainer):
    def validate(self, split):
        self.model.eval()
        val_criterion = self.criterion  # torch.nn.CrossEntropyLoss()

        test_loss = DistributedMetric("test_loss")
        test_top1 = DistributedMetric("test_top1")

        with torch.no_grad():
            # This split evaluates the performance of the model on the testing set
            if split == "test":
                with tqdm(
                    total=len(self.data_loader["test"]),
                    desc="Test",
                    disable=dist.rank() > 0 or config.ray_tune,
                ) as t:
                    for images, labels in self.data_loader["test"]:
                        images, labels = images.cuda(), labels.cuda()
                        # compute output
                        output = self.model(images)
                        loss = val_criterion(output, labels)
                        test_loss.update(loss, images.shape[0])
                        acc1 = accuracy(output, labels, topk=(1,))[0]
                        test_top1.update(acc1.item(), images.shape[0])

                        t.set_postfix(
                            {
                                "loss": test_loss.avg.item(),
                                "top1": test_top1.avg.item(),
                                "batch_size": images.shape[0],
                                "img_size": images.shape[2],
                            }
                        )
                        t.update()

            # This split is performed to compute neuron velocities
            elif split == "val":
                with tqdm(
                    total=len(self.data_loader["val"]),
                    desc="Validate",
                    disable=dist.rank() > 0 or config.ray_tune,
                ) as t:
                    for images, labels in self.data_loader["val"]:
                        images, labels = images.cuda(), labels.cuda()
                        # compute output
                        output = self.model(images)
                        loss = val_criterion(output, labels)

                        test_loss.update(loss, images.shape[0])
                        acc1 = accuracy(output, labels, topk=(1,))[0]
                        test_top1.update(acc1.item(), images.shape[0])

                        t.set_postfix(
                            {
                                "loss": test_loss.avg.item(),
                                "top1": test_top1.avg.item(),
                                "batch_size": images.shape[0],
                                "img_size": images.shape[2],
                            }
                        )
                        t.update()

        return {
            "test/top1": test_top1.avg.item(),
            "test/loss": test_loss.avg.item(),
        }

    def train_one_epoch(self, epoch):
        self.model.train()
        self.data_loader["train"].sampler.set_epoch(epoch)

        train_loss = DistributedMetric("train_loss")
        train_top1 = DistributedMetric("train_top1")

        with tqdm(
            total=len(self.data_loader["train"]),
            desc="Train Epoch #{}".format(epoch + 1),
            disable=dist.rank() > 0 or config.ray_tune,
        ) as t:
            for _, (images, labels) in enumerate(self.data_loader["train"]):
                images, labels = images.cuda(), labels.cuda()
                self.optimizer.zero_grad()

                output = self.model(images)
                loss = self.criterion(output, labels)
                # backward and update
                loss.backward()
                for k in self.grad_mask:
                    zero_gradients(self.model, k, self.grad_mask[k])

                self.optimizer.step()

                # after one step
                train_loss.update(loss, images.shape[0])
                acc1 = accuracy(output, labels, topk=(1,))[0]
                train_top1.update(acc1.item(), images.shape[0])

                t.set_postfix(
                    {
                        "loss": train_loss.avg.item(),
                        "top1": train_top1.avg.item(),
                        "batch_size": images.shape[0],
                        "img_size": images.shape[2],
                        "lr": self.optimizer.param_groups[0]["lr"],
                    }
                )
                t.update()

                # after step (NOTICE that lr changes every step instead of epoch)
                self.lr_scheduler.step()

        return_dict = {
            "train/top1": train_top1.avg.item(),
            "train/loss": train_loss.avg.item(),
            "train/lr": self.optimizer.param_groups[0]["lr"],
        }
        return return_dict
