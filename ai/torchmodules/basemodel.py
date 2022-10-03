import math
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import get_default_path


class BaseModel(nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self._name = name
        self._loss_func = None
        self._optimizer = None
        self._scheduler = None

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def calc_loss(self, x_inputs=None, y_targets=None):
        if issubclass(type(x_inputs), DataLoader):
            return self._calc_iterable_loss(x_inputs)

        if y_targets is None:
            raise Exception(
                "First argument is not a loader and did not provide targets"
            )

        return self._calc_batch_loss(x_inputs, y_targets)

    def _calc_iterable_loss(self, data_loader, print_every=250):

        val_loss = 0
        scaler = torch.cuda.amp.GradScaler()
        start_time = time.perf_counter()
        for batch_num, data in enumerate(data_loader):
            x, y = data
            loss = self.calc_loss(x.to(self._device, non_blocking=True), y.to(self._device, non_blocking=True))
            scaler.scale(loss)
            val_loss += loss.item()

            if batch_num != 0 and batch_num % print_every == 0:
                sec_per_batch = (time.perf_counter() - start_time) / print_every
                batch_per_sec = 1 / sec_per_batch
                rem_time = sec_per_batch * (len(data_loader) - batch_num)
                cur_loss = val_loss / batch_num
                ppl = math.exp(cur_loss)
                print(
                    f"{batch_num:5d}/{len(data_loader):5d} batches | "
                    f"batch/sec {batch_per_sec:5.2f} | "
                    f"rem mins {rem_time/60:5.0f} | "
                    f"loss {cur_loss:5.4f} | ppl {ppl:8.4f}"
                )
                start_time = time.perf_counter()

        return val_loss / batch_num

    def _calc_batch_loss(self, x_inputs, y_targets):
        outputs = self(x_inputs)
        return self._loss_func(outputs, y_targets)

    def get_loss_func(self):
        if not self._loss_func:
            raise Exception("Model did not set a loss function")
        return self._loss_func

    def get_optimizer(self):
        if not self._optimizer:
            raise Exception("Model did not set an optimizer")
        return self._optimizer

    def get_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        if self._scheduler is not None:
            return self._scheduler

        optimizer = self.get_optimizer()
        return torch.optim.lr_scheduler.StepLR(optimizer, sys.maxsize, gamma=1.0)

    def load_best(self, strict=True):
        model_path = get_default_path(self.get_save_dir(), "best")
        self.load_state_dict(torch.load(model_path), strict=strict)

    def save_as_best(self):
        model_path = get_default_path(self.get_save_dir(), "best")
        torch.save(self.state_dict(), model_path)

    def get_name(self) -> str:
        return self._name

    def get_save_dir(self) -> str:
        return get_default_path(f"model/{self._name}")
