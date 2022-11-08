import typing

import torch


class WarmupLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, target_rate, warmup_period, last_epoch=0, start_rate=1e-10, verbose=False):
        self._start_rate = start_rate
        self._target_rate = target_rate
        self._warmup_period = warmup_period

        super(WarmupLR, self).__init__(optimizer, last_epoch, verbose)
        self.optimizer = optimizer

    def state_dict(self):
        state_dict = {key: value for key, value in self.__dict__.items() if key != "_optimizer"}

        return state_dict

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
        self.step(self.last_epoch)

    def get_lr(self):
        percent = min(self.last_epoch / self._warmup_period, 1.0)
        return [percent * self._target_rate + (1 - percent) * self._start_rate]


class ImprovedCyclicLR(torch.optim.lr_scheduler.CyclicLR):
    def __init__(
        self,
        *args,
        mode="improved_exp_range",
        scale_mode: str = "cycle",
        last_epoch: int = None,
        cycle_offset: int = 0,
        scale_fn=None,
        **kwargs
    ) -> None:

        # Optionally begin partway through a cycle
        self.cycle_offset = cycle_offset

        if mode == "improved_exp_range":
            scale_fn = self._improved_exp_scale_fn
            scale_mode = "cycle"

        super().__init__(*args, scale_fn=scale_fn, scale_mode=scale_mode, **kwargs)
        if last_epoch is not None:
            self.last_epoch = self._adjust_epoch(last_epoch)
            self.step(self.last_epoch)

        # Don't force exp to iteration mode. It's unclear why the base class does that--it makes the mode non-cyclitic
        # unless gamma is incredibly close to 1.0
        self.scale_mode = scale_mode

    def _improved_exp_scale_fn(self, x):
        # The default exp scale func starts at x=1 resulting in a skipped cycle
        return self.gamma ** max(x - 1, 0)

    def state_dict(self):
        state_dict = super().state_dict()
        state_dict.update({"cycle_offset": self.cycle_offset})

        return state_dict

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
        self.step(self.last_epoch)

    def step(self, epoch=None):
        adjusted_epoch = self._adjust_epoch(epoch)
        # print(f"Stepping {adjusted_epoch}, pre {self.last_epoch}")
        super().step(adjusted_epoch)

    def _adjust_epoch(self, epoch):
        if epoch is None:
            return None
        return self.cycle_offset + max(epoch, -1)


def make_cyclic_with_warmup(
    optimizer,
    epoch_batches,
    max_lr,
    min_lr_divisor=10,
    gamma=0.75,
    step_size_up_epoch_mul=0.5,
    warmup_period_epoch_mul=1,
    last_epoch=None,
    **kwargs
):
    step_size_up = int(epoch_batches * step_size_up_epoch_mul)
    warmup_period = int(epoch_batches * warmup_period_epoch_mul)

    base_lr = max_lr / min_lr_divisor

    cycle_offset = 0
    skip_warmup = warmup_period_epoch_mul == 0 or (last_epoch is not None and last_epoch >= warmup_period)
    if skip_warmup:
        cycle_offset = 0 if last_epoch is None else last_epoch - warmup_period

    cyclic_lr = ImprovedCyclicLR(
        optimizer,
        base_lr=base_lr,
        max_lr=max_lr,
        step_size_up=step_size_up,
        cycle_offset=cycle_offset,
        mode="improved_exp_range",
        last_epoch=last_epoch,
        gamma=gamma,
        **kwargs,
    )

    if skip_warmup:
        print("Skipping warmup")
        return cyclic_lr

    warmup = WarmupLR(optimizer, base_lr, warmup_period, last_epoch=last_epoch)
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup, cyclic_lr], last_epoch=last_epoch, milestones=[warmup_period]
    )
