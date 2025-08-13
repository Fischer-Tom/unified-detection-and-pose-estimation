import os
import torch
import torch.distributed as dist

def all_gather_obj(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    data_list = [None] * world_size
    dist.all_gather_object(data_list, data)
    return data_list

def all_gather_tensor_varying_batch(tensor: torch.Tensor) -> torch.Tensor:
    world_size = dist.get_world_size()

    # Step 1: Gather sizes
    local_size = torch.tensor([tensor.shape[0]], device=tensor.device)
    sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(sizes, local_size)
    sizes = [s.item() for s in sizes]
    max_size = max(sizes)

    # Step 2: Pad to max size along batch dim (dim=0)
    pad_size = list(tensor.shape)
    pad_size[0] = max_size - tensor.shape[0]
    padding = torch.zeros(pad_size, dtype=tensor.dtype, device=tensor.device)
    padded = torch.cat([tensor, padding], dim=0)

    # Step 3: All gather
    gathered = [torch.empty_like(padded) for _ in range(world_size)]
    dist.all_gather(gathered, padded)

    # Step 4: Unpad/slice
    tensors = [g[:size] for g, size in zip(gathered, sizes)]
    return torch.cat(tensors, dim=0)

def all_gather_tensor_dict(in_dict, keys=None):
    world_size = get_world_size()
    out_dict = {}
    if keys is None:
        keys = list(in_dict.keys())
    if world_size == 1:
        for k in keys:
            out_dict[k] = in_dict[k]
        return out_dict
    else:
        for k in keys:
            v = in_dict[k]
            if isinstance(v, torch.Tensor):
                out_dict[k] = all_gather_tensor_varying_batch(v)
            else:
                raise TypeError(f"Error: {k} is not a tensor, but {type(v)}. Only tensors are supported for all_gather_tensor_dict.")
        return out_dict


def reduce_dict(input_dict, reduce=True, to_gather=None):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.inference_mode():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        keys = list(input_dict.keys()) if to_gather is None else to_gather

        for k in sorted(keys):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        if reduce:
            dist.all_reduce(values)
            values /= world_size
            reduced_dict = {k: v for k, v in zip(names, values)}
        else:
            gathered_values = [torch.zeros_like(values) for _ in range(world_size)]
            dist.all_gather(gathered_values, values)
            reduced_dict = {k: torch.cat([gv[i] for gv in gathered_values], dim=0) for i, k in enumerate(names)}
    return reduced_dict

def gather_tensor(features):
    rank = get_rank()
    world_size = get_world_size()

    gathered = [torch.zeros_like(features) for _ in range(world_size)] if rank == 0 else None
    dist.gather(features, gather_list=gathered, dst=0)
    if rank == 0:
        all_features = torch.cat(gathered, dim=0)
        return all_features
    else:
        return None

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def broadcast_object(obj, src=0):
    if not is_dist_avail_and_initialized():
        return
    dist.broadcast(obj, src)


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0

def dprint(*args, **kwargs):
    """
    Print only if the current process is the main process.
    """
    if is_main_process():
        print(*args, **kwargs)


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

def init_distributed_mode(cfg):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        cfg.rank = int(os.environ["RANK"])
        cfg.world_size = int(os.environ["WORLD_SIZE"])
        cfg.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        cfg.rank = int(os.environ["SLURM_PROCID"])
        cfg.world_size = int(os.environ["SLURM_NTASKS"])
        cfg.gpu = cfg.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode", flush=True)
        return False

    distributed = True

    torch.cuda.set_device(cfg.gpu)
    dist_backend = "nccl"
    print(f"| distributed init (rank {cfg.rank}): {cfg.dist_url}", flush=True)
    torch.distributed.init_process_group(
        backend=dist_backend, init_method=cfg.dist_url, world_size=cfg.world_size, rank=cfg.rank
    )
    torch.distributed.barrier(device_ids=[cfg.gpu])
    setup_for_distributed(cfg.rank == 0)
    return distributed