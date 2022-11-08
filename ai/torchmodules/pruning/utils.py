import ai.torchmodules as torchmodules
import ai.torchmodules.utils as torchutils
import torch
import torch_pruning as tp


def prune_loss_limited(model, pruner, data_loader, initial_loss=None, max_loss_increase=1.1, max_steps=10):
    if initial_loss is None:
        initial_loss = estimate_validation(model, data_loader)
    val = initial_loss
    val_limit = val * max_loss_increase

    ori_size = tp.utils.count_params(model)
    print(f"Start validation is {val:.4f}. Pruning until {val_limit:.4f}")
    torchmodules.save_model(model, model.name, "best_pruned_last")
    for idx in range(max_steps):
        model.cpu()
        pruner.step()
        torchutils.torch_garbage_collect()
        model.cuda()
        num_params = tp.utils.count_params(model) / 1e6
        print(f"Params: {ori_size / 1e6:.2f} M => {num_params:.2f} M")

        val = estimate_validation(model, data_loader)
        if val < val_limit:
            print(f"Val was {val:.4f}, saving checkpoint")
            torchmodules.save_model(model, model.name, "best_pruned_last")
        else:
            print(f"Val was {val:.4f}. Stopping and loading last checkpoint")
            break
    return torchmodules.load_model(model.name, "best_pruned_last")


def estimate_validation(
    model, data_loader, max_batches=1000, low_change_thresh=0.0025, req_low_change=5, check_freq=10, verbose=False
):
    model = model.cuda()
    model.train(False)

    data_loader.dataset.clear()
    torchutils.torch_garbage_collect()
    prev_loss = 0
    con_low_change = 0
    print_freq = 10
    highest_reset = 0
    with torch.no_grad():
        loss = 0

        for idx, data in enumerate(data_loader):
            if issubclass(type(data), dict):
                torchutils.dict_to_device(data)
                loss += model.calc_loss(**data)
            else:
                data = (data[0].to(model.device, non_blocking=True), data[1].to(model.device, non_blocking=True))
                loss += model.calc_loss(*data)

            if idx % check_freq == 0:
                curr_loss = loss / (idx + 1)
                change_percent = abs(curr_loss - prev_loss) / prev_loss

                if change_percent < low_change_thresh:
                    con_low_change += 1
                else:
                    highest_reset = max(con_low_change, highest_reset)
                    con_low_change = 0
                    prev_loss = curr_loss

                if verbose and (idx % print_freq == 0 or con_low_change == req_low_change):
                    if prev_loss != 0:
                        print(f"Loss is {curr_loss:.4f} changing {change_percent:.4f}% from baseline {prev_loss:.4f}.")
                        print(f"Consecutive low changes {con_low_change}/{req_low_change}")
                    else:
                        print(f"First loss {curr_loss:.4f}")
                    print(f"Highest reset was {highest_reset}")

                if con_low_change == req_low_change:
                    print(f"Highest reset was {highest_reset}")
                    break

            if idx >= max_batches:
                break
        return (loss / idx + 1).cpu().item()
