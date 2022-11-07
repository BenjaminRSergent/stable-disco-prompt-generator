import datetime
import math
import time

import ai.torchmodules.utils as torchutils
import torch
from ai.torchmodules import BaseModel
from torch.utils.data import DataLoader

from utils import get_default_path


class TorchTrainer:
    def __init__(self, batch_size: int, patience: int) -> None:
        self._batch_size = batch_size
        self._patience = patience

    def train_early_stop(
        self,
        model: BaseModel,
        training_loader: DataLoader,
        validation_loader: DataLoader,
        best_validation: float = None,
        out_dir: str = None,
        patience=2,
        max_epochs=100,
        num_to_acc=1,
        print_every=1000,
        print_learning_rate=False
    ) -> float:
        new_best_val_loss = best_validation
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if not out_dir:
            out_dir = model.get_save_dir()

        since_improvement = 0


        print("Starting training")
        for epoch in range(max_epochs):
            print(f"Starting epoch {epoch}")
            model.train(True)
            torchutils.torch_garbage_collect()
            loss = self._train_epoch(
                model, training_loader, num_to_acc, print_every=print_every, print_learning_rate=print_learning_rate
            )

            print(f"Epoch loss: {loss}")
            model.train(False)
            if "clear" in dir(training_loader.dataset):
                training_loader.dataset.clear()
            torchutils.torch_garbage_collect()

            model_path = get_default_path(
                model.get_save_dir(), f"{timestamp}_epoch_{epoch}_tmp"
            )
            torch.save(model.state_dict(), model_path)

            print("Calcing val loss")
            with torch.autocast(device_type="cuda"):
                val_loss = model.calc_loss(validation_loader)
            print(f"val_loss: {val_loss}")

            if "clear" in dir(validation_loader.dataset):
                validation_loader.dataset.clear()
            torchutils.torch_garbage_collect()

            if not new_best_val_loss or val_loss < new_best_val_loss:
                print("Model improved, saving")
                since_improvement = 0
                new_best_val_loss = val_loss
                model_path = get_default_path(
                    model.get_save_dir(), f"{timestamp}_epoch_{epoch}"
                )
                torch.save(model.state_dict(), model_path)
            else:
                since_improvement += 1
                if since_improvement > patience:
                    print("Early stopping triggered")
                    break

        return new_best_val_loss

    def _train_epoch(
        self,
        model: BaseModel,
        data_loader: DataLoader,
        num_to_acc: int,
        print_every: int,
        print_learning_rate=False,
        clip_val=0.5
    ):
        total_loss = 0
        running_loss = 0
        optimizer = model.get_optimizer()
        scaler = torch.cuda.amp.GradScaler()
        start_time = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        scheduler = model.get_scheduler()

        device = model.get_device()
        for idx, data in enumerate(data_loader):
            

            with torch.autocast(device_type="cuda"):
                if issubclass(type(data), dict):
                    torchutils.dict_to_device(data, device)
                    loss = model.calc_loss(**data) / num_to_acc
                else:
                    for data_idx in range(len(data)):
                        data[data_idx] = data[data_idx].to(device, non_blocking=True)
                    loss = model.calc_loss(*data) / num_to_acc
            # Scales the loss, and calls backward()
            # to create scaled gradients
            scaler.scale(loss).backward()

            if num_to_acc < 2 or (idx + 1) % num_to_acc == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
                scaler.step(optimizer)
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                if scheduler:
                    scheduler.step()

            total_loss += loss.item() * num_to_acc
            running_loss += loss.item() * num_to_acc

            if (idx+1) % print_every == 0:
                sec_per_batch = (time.perf_counter() - start_time) / print_every
                batch_per_sec = 1 / sec_per_batch
                rem_time = sec_per_batch * (len(data_loader) - idx)
                cur_loss = running_loss / print_every
                ppl = math.exp(cur_loss)
                print(
                    f"{idx:5d}/{len(data_loader):5d} batches | "
                    f"batch/sec {batch_per_sec:5.2f} | "
                    f"rem mins {rem_time/60:5.0f} | "
                    f"loss {cur_loss:5.6f} | ppl {ppl:8.4f}"
                )
                if print_learning_rate:
                    print(f"Learning rate: {scheduler.get_last_lr()}")

                running_loss = 0
                start_time = time.perf_counter()


        return total_loss / idx
