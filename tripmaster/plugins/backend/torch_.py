
from tripmaster.core.components.backend import TMOptimizerBehaviors, TMBasicTensorOperations, TMTypes, TMBackend, \
    TMBasicModuleOperations, TMBasicDeviceOperations
from tripmaster.core.components.operator.strategies.distributed import TMDistributedStrategy

import logging

logger = logging.getLogger(__name__)
import random
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data import Sampler
import math
import numpy as np
from typing import Iterator, Iterable, Optional, Sequence, List, TypeVar, Generic, Sized, Union
from copy import deepcopy
import multiprocessing
from multiprocessing import Process


# __all__ = [
#     "FillBatchSampler",
#     "FillRandomSampler",
#     "Sampler",
#     "SequentialSampler",
#     "SubsetRandomSampler",
#     "WeightedRandomSampler",
# ]

T_co = TypeVar('T_co', covariant=True)


from collections.abc import Sequence, Mapping


def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def reduce_mean(tensor, nprocs):
    """

    Args:
        tensor:
        nprocs:

    Returns:

    """
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt

class TorchNoDistributedStrategy(TMDistributedStrategy):
    """
    TorchNoDistributedStrategy
    """
    Name = "no"

    def __init__(self, operator, world_size, use_gpu):
        super().__init__(operator, world_size, use_gpu)

        if world_size > 1 and use_gpu:
            logger.warning(
                f"You have request to use multiple gpus, but with Single trainer. Only one of the gpus will be used")

    def run(self, func, train_data_streams, runtime_options):
        """
        train_impl
        """
        return func(0, self.operator, train_data_streams, runtime_options)

    def init(self, local_rank):
        """
        init_train_loop
        """

        if self.use_gpu:
            device = "cuda:" + str(local_rank)
        else:
            device = "cpu"

        self.operator.machine.to(device)

    def sync_loss(self, loss):
        """
        sync_loss
        """
        return loss


class TorchDPMachine(torch.nn.DataParallel):

    def __init__(self, machine, *args, **kwargs):

        super().__init__(machine, *args, **kwargs)

    def __getattr__(self, name):

        try:
            return torch.nn.DataParallel.__getattr__(self, name)
        except AttributeError as e:
            if hasattr(self.module, name):
                return getattr(self.module, name)
            else:
                logger.error(f"missing attribute {name}")
                raise e



class TorchDataParallelStrategy(TMDistributedStrategy):
    """
    ChaosdDPTrainer
    """

    Name = "dp"

    def run(self, func, train_data_streams, runtime_options):
        """
        train_impl
        """

        return func(0, self.operator, train_data_streams, runtime_options)

    def init(self, local_rank):
        """
        init_train_loop
        """

        self.distributed = False
        if self.use_gpu:
            device = "cuda:" + str(local_rank)
        else:
            device = "cpu"

        self.operator.machine = TorchDPMachine(self.operator.machine)
        self.operator.machine.to(device)


    def sync_loss(self, loss):
        """
        sync_loss
        """

        return loss


class TorchDDPMachine(torch.nn.parallel.DistributedDataParallel):
    """
    TorchDDPMachine
    """

    def __init__(self, machine, *args, **kwargs):

        super().__init__(machine, *args, **kwargs)

    def __getattr__(self, name):

        try:
            return torch.nn.parallel.DistributedDataParallel.__getattr__(self, name)
        except AttributeError as e:
            if hasattr(self.module, name):
                return getattr(self.module, name)
            else:
#                logger.error(f"missing attribute {name}")
                raise e


class TorchDistributedDataParallelStrategy(TMDistributedStrategy):
    """
    TorchDistributedDataParallelStrategy
    """

    Name = "ddp"


    def __init__(self, operator, world_size, use_gpu):
        super().__init__(operator, world_size, use_gpu)

        if use_gpu:
            if world_size > 1:
                logger.info("Using DDP trainer for multiple gpu")
            else:
                logger.warning(
                    f"You have request to use DDP trainer, but you have configured to use only one gpu.")
        else:
            logger.warning(
                f"You have request to use DDP trainer, but you have configured to use no gpu. Only for test purpose")

    def run(self, func, train_data_streams, runtime_options):
        """
        train_impl
        """

        import port_for
        self.distributed_port = port_for.select_random()

        mp.spawn(func, nprocs=self.world_size, args=(self.operator, train_data_streams, runtime_options))

        return None

    def init(self, local_rank):
        """
        init_train_loop
        """

        self.distributed = True

        if self.use_gpu:
            device = "cuda:" + str(local_rank)
            backend = "nccl"
            device_ids = [local_rank]
        else:
            device = "cpu"
            backend = "gloo"
            device_ids = None

        dist.init_process_group(backend=backend, init_method=f'tcp://127.0.0.1:{self.distributed_port}',
                                world_size=self.world_size,
                                rank=local_rank)

        # When using a single GPU per process and per
        # DistributedDataParallel, we need to divide the batch size
        # ourselves based on the total number of GPUs we have
        self.operator.machine.to(device)

        for name, opt in self.operator.optimization_strategy.optimizer.items():
            optimizer_to(opt, device)
            # self.operator.optimization_strategy.optimizer[name] = opt.to(device)
        self.operator.machine = TorchDDPMachine(self.operator.machine, device_ids=device_ids,
                                                find_unused_parameters=True)


    def sync_loss(self, loss):
        """
        sync_loss
        """

        torch.distributed.barrier()

        reduced_loss = reduce_mean(loss, dist.get_world_size())

        return reduced_loss


class TorchOptimizerBehaviors(TMOptimizerBehaviors):

    @classmethod
    def optimizer_state_dict(cls, optimizer):

        T = TorchBasicTensorOperations
        np_opt = {'param_groups': optimizer.state_dict()['param_groups']}
        state = optimizer.state_dict()['state']
        np_s = {k: {kk: T.to_numpy(T.to_device(vv, "cpu")) if isinstance(vv, T.Tensor) else vv
                    for kk, vv in v.items()} for k, v in state.items()}
        np_opt["state"] = np_s
        return np_opt

    @classmethod
    def load_optimizer_state(cls, optimizer, state_dict):

        T = TorchBasicTensorOperations
        M = TorchBasicModuleOperations
        opt_s = state_dict["state"]
        state = {k: {kk: T.to_device(T.to_tensor(vv), "cpu")
                 if isinstance(vv, np.ndarray) else vv for kk, vv in v.items()}
                 for k, v in opt_s.items()}
        state_dict["state"] = state
        M.load_state_dict(optimizer, state_dict)

    @classmethod
    def set_train_mode(self, machine):
        machine.train()

    @classmethod
    def set_inference_mode(self, machine):
        machine.eval()

    no_grad = torch.no_grad

    @classmethod
    def clear_grad(self, optimizer):

        if isinstance(optimizer, Sequence):
            for optim in optimizer:
                optim.zero_grad()
        elif isinstance(optimizer, Mapping):
            for name, optim in optimizer.items():
                self.clear_grad(optim)
        else:
            optimizer.zero_grad()

    @classmethod
    def backward_loss(self, loss):
        loss.backward()

    @classmethod
    def optimizer_step(self, optimizer):
        if isinstance(optimizer, Sequence):
            for optim in optimizer:
                optim.step()
        elif isinstance(optimizer, Mapping):
            for name, optim in optimizer.items():
                self.optimizer_step(optim)
        else:
            optimizer.step()

    @classmethod
    def lrscheduler_step(self, lr_scheduler):
        if isinstance(lr_scheduler, Sequence):
            for scheduler in lr_scheduler:
                scheduler.step()
        elif isinstance(lr_scheduler, Mapping):
            for name, scheduler in lr_scheduler.items():
                self.lrscheduler_step(scheduler)
        else:
            lr_scheduler.step()

    @classmethod
    def clip_gradient_norm(self, machine, clip_norm_threshold):
        torch.nn.utils.clip_grad_norm_(machine.parameters(), clip_norm_threshold) 


    @classmethod
    def create_optimization_components(self, parameters,
                                   optimizer_type, optimizer_params,
                                   lr_scheduler_type, lr_scheduler_params,
                                   gradient_clip_val):

        if "lr" in lr_scheduler_params and "lr" not in optimizer_params:
            optimizer_params.lr = lr_scheduler_params.lr
            del lr_scheduler_params["lr"]
        if "learning_rate" in lr_scheduler_params and "learning_rate" not in optimizer_params:
            optimizer_params.lr = lr_scheduler_params.learning_rate
            del lr_scheduler_params["learning_rate"]

        optimizer = optimizer_type(params=parameters, **optimizer_params)

        lr_scheduler = lr_scheduler_type(optimizer=optimizer, **lr_scheduler_params)

        return optimizer, lr_scheduler


class TorchTypes(TMTypes):

    Bool = torch.bool
    Int8 = torch.int8
    Int16 = torch.int16
    Int32 = torch.int32
    Int64 = torch.int64
    Float16 = torch.float16
    Float32 = torch.float32
    Float64 = torch.float64
    Float = torch.float32

from more_itertools import chunked

class TorchBasicTensorOperations(TMBasicTensorOperations):

    Tensor = torch.Tensor

    @classmethod
    def is_tensor(self, x):
        return isinstance(x, torch.Tensor)

    @classmethod
    def pad(self, input, pad, value=0.0):

        return torch.nn.functional.pad(input, pad, mode="constant", value=value)

    @classmethod
    def to_tensor(self, x, dtype=None):
        return torch.tensor(x, dtype=dtype)

    @classmethod
    def stack(self, list, dim):
        return torch.stack(list, dim=dim)

    @classmethod
    def cast(self, x, type):
        return x.to(type)

    @classmethod
    def to_numpy(self, x):
        return x.numpy()

    @classmethod
    def to_device(self, x, device):
        if device.startswith("gpu"):
            device = device.replace("gpu", "cuda")
        return x.to(device)

    @classmethod
    def device(cls, x):
        return x.device.type

    @classmethod
    def shape(self, x):
        return x.shape

class TorchBasicDeviceOperations(TMBasicDeviceOperations):

    @classmethod
    def is_device(self, x):
        raise NotImplementedError()

    @classmethod
    def set_device(cls, device):
        if device.startswith("cuda"):
            torch.cuda.set_device(device)

    @classmethod
    def cuda_available(cls):
        return torch.cuda.is_available()
    
    @classmethod
    def device_prefix(cls):
        return "cuda"



class TorchBasicModuleOperations(TMBasicModuleOperations):

    Module = torch.nn.Module
    ModuleList = torch.nn.ModuleList
    ModuleDict = torch.nn.ModuleDict

    @classmethod
    def load_state_dict(cls, module: torch.nn.Module, state_dict):
        module.load_state_dict(state_dict)

    @classmethod
    def to_device(cls, module: torch.nn.Module, device):
        return module.to(device)


class DistributedBatchSampler(torch.utils.data.BatchSampler):
    """ `BatchSampler` wrapper that distributes across each batch multiple workers.

    Args:
        batch_sampler (torch.utils.data.sampler.BatchSampler)
        num_replicas (int, optional): Number of processes participating in distributed training.
        rank (int, optional): Rank of the current process within num_replicas.

    Example:
        >>> from torch.utils.data.sampler import BatchSampler
        >>> from torch.utils.data.sampler import SequentialSampler
        >>> sampler = SequentialSampler(list(range(12)))
        >>> batch_sampler = BatchSampler(sampler, batch_size=4, drop_last=False)
        >>>
        >>> list(DistributedBatchSampler(batch_sampler, num_replicas=2, rank=0))
        [[0, 2], [4, 6], [8, 10]]
        >>> list(DistributedBatchSampler(batch_sampler, num_replicas=2, rank=1))
        [[1, 3], [5, 7], [9, 11]]
    """

    def __init__(self, batch_sampler, **kwargs):
        self.batch_sampler = batch_sampler
        self.kwargs = kwargs
        self.epoch = 0

    def __iter__(self):
        for batch in self.batch_sampler:
            distributed_sampler = torch.utils.data.DistributedSampler(batch, **self.kwargs)
            distributed_sampler.set_epoch(self.epoch)
            yield [batch[x] for x in distributed_sampler]

    #            yield list(DistributedSampler(batch, **self.kwargs))

    def __len__(self):
        return len(self.batch_sampler)

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Arguments:
            epoch (int): Epoch number.
        """
        self.batch_sampler.set_epoch(epoch)


class TMTorchBackend(TMBackend):

    Name = "torch"

    NoParallelStrategy = TorchNoDistributedStrategy
    DataParallelStrategy = TorchDataParallelStrategy
    DistributedDataParallelStrategy = TorchDistributedDataParallelStrategy

    OptimizerBehaviors = TorchOptimizerBehaviors

    DataLoader = torch.utils.data.DataLoader
    Dataset = torch.utils.data.Dataset
    IterableDataset = torch.utils.data.IterableDataset

    Sampler = torch.utils.data.Sampler
    BatchSampler = torch.utils.data.BatchSampler
    SubsetRandomSampler = torch.utils.data.SubsetRandomSampler
    SequenceSampler = torch.utils.data.SequentialSampler
    RandomSampler = torch.utils.data.RandomSampler
    SequentialSampler = torch.utils.data.SequentialSampler
    DistributedSampler = torch.utils.data.DistributedSampler
    DistributedBatchSampler = DistributedBatchSampler

    Types = TorchTypes
    BasicTensorOperations = TorchBasicTensorOperations
    BasicModuleOperations = TorchBasicModuleOperations
    BasicDeviceOperations = TorchBasicDeviceOperations
    Dist = torch.distributed
