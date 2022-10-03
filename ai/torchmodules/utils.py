import gc

import torch


def get_default_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def generate_square_subsequent_mask(size: int, device=None) -> torch.Tensor:
    if device is None:
        device = get_default_device()

    mask = (
        torch.triu(torch.ones(size, size, device=device, dtype=torch.float)) == 1
    ).transpose(0, 1)
    return mask.masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))


def get_split_idxs(total_data, percent):
    second_data_start = int(total_data * (1 - percent))
    return (0, second_data_start), (second_data_start, total_data)


def torch_garbage_collect():
    gc.collect()
    torch.cuda.empty_cache()


def unravel_torch_idx(idx, cols, device=None):
    if device is None:
        device = get_default_device()

    idx_one = torch.div(idx, cols, rounding_mode="trunc")
    idx_two = torch.remainder(idx, cols)
    return torch.stack((idx_one, idx_two), dim=1).long()

def refresh_cuda_memory():
    """
    Re-allocate all cuda memory to help alleviate fragmentation
    """
    # Run a full garbage collect first so any dangling tensors are released
    torch_garbage_collect()

    # Then move all tensors to the CPU
    locations = {}
    for obj in gc.get_objects():
        if not isinstance(obj, torch.Tensor):
            continue

        locations[obj] = obj.device
        obj.data = obj.data.cpu()
        if isinstance(obj, torch.nn.Parameter) and obj.grad is not None:
            obj.grad.data = obj.grad.cpu()

    # Now empty the cache to flush the allocator
    torch.cuda.empty_cache()

    # Finally move the tensors back to their associated GPUs
    for tensor, device in locations.items():
        tensor.data = tensor.to(device)
        if isinstance(tensor, torch.nn.Parameter) and tensor.grad is not None:
            tensor.grad.data = tensor.grad.to(device)
