from tripmaster.core.components.backend import TMOptimizerBehaviors, TMBasicTensorOperations, TMTypes, TMBackend, \
    TMBasicModuleOperations, TMBasicDeviceOperations
from tripmaster.core.components.operator.strategies.distributed import TMDistributedStrategy

import logging

logger = logging.getLogger(__name__)
import random
import math
import numpy as np
from typing import Any, Iterator, Iterable, Optional, Sequence, List, TypeVar, Generic, Sized, Union
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
    pass

def reduce_mean(tensor, nprocs):
    """

    Args:
        tensor:
        nprocs:

    Returns:

    """
    raise NotImplementedError()


class CPUNoDistributedStrategy(TMDistributedStrategy):
    """
    CPUNoDistributedStrategy
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

        device = self.operator.device(local_rank)

        self.operator.machine.to(device)

    def sync_loss(self, loss):
        """
        sync_loss
        """
        return loss


class CPUDPMachine:

    def __init__(self, machine, *args, **kwargs):

        # super().__init__(machine, *args, **kwargs)
        pass




class CPUDataParallelStrategy(TMDistributedStrategy):
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

        self.operator.machine = self.operator.machine

    def sync_loss(self, loss):
        """
        sync_loss
        """

        return loss


class TorchDDPMachine:
    """
    TorchDDPMachine
    """

    def __init__(self, machine, *args, **kwargs):

        super().__init__(machine, *args, **kwargs)


class CPUDistributedDataParallelStrategy(TMDistributedStrategy):
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
        raise NotImplementedError()

    def init(self, local_rank):
        """
        init_train_loop
        """

        raise NotImplementedError()
    def sync_loss(self, loss):
        """
        sync_loss
        """

        # distributed.barrier()

        # reduced_loss = reduce_mean(loss, dist.get_world_size())

        # return reduced_loss

        raise NotImplementedError()


class CPUOptimizerBehaviors(TMOptimizerBehaviors):

    @classmethod
    def optimizer_state_dict(cls, optimizer):

        T = CPUBasicTensorOperations
        np_opt = {'param_groups': optimizer.state_dict()['param_groups']}
        state = optimizer.state_dict()['state']
        np_s = {k: {kk: T.to_numpy(T.to_device(vv, "cpu")) if isinstance(vv, T.Tensor) else vv
                    for kk, vv in v.items()} for k, v in state.items()}
        np_opt["state"] = np_s
        return np_opt

    @classmethod
    def load_optimizer_state(cls, optimizer, state_dict):

        T = CPUBasicTensorOperations
        M = CPUBasicModuleOperations
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

    no_grad = None

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
        raise NotImplementedError()

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


class CPUTypes(TMTypes):
    Bool = bool
    Int8 = np.int8
    Int16 = np.int16
    Int32 = np.int32
    Int64 = np.int64
    Float16 = np.float16
    Float32 = np.float32
    Float64 = np.float64
    Float = np.float32


from more_itertools import chunked


class TensorOpMeta(type):
    def __getattr__(cls, name):
        return getattr(np, name)


class CPUBasicTensorOperations(TMBasicTensorOperations, metaclass=TensorOpMeta):
    Tensor = np.ndarray

    @classmethod
    def zeros(cls, *args, dtype=None, device=None):
        return np.zeros(args, dtype=dtype, device=device)

    @classmethod
    def ones(cls, *args, dtype=None, device=None):
        return np.ones(*args, dtype=dtype, device=device)

    @classmethod
    def all(cls, a):
        return np.all(a)

    @classmethod
    def any(cls, a):
        return np.any(a)

    @classmethod
    def logical_and(cls, a, b):
        return np.logical_and(a, b)

    @classmethod
    def logical_or(cls, a, b):
        return np.logical_or(a, b)

    @classmethod
    def logical_not(cls, a):
        return np.logical_not(a)

    @classmethod
    def is_tensor(self, x):
        return isinstance(x, np.Tensor)

    @classmethod
    def pad(self, input, pad, value=0.0):
        return np.nn.functional.pad(input, pad, mode="constant", value=value)

    @classmethod
    def to_tensor(self, x, dtype=None):
        return np.tensor(x, dtype=dtype)

    @classmethod
    def stack(self, list, dim):
        return np.stack(list, dim=dim)

    @classmethod
    def cast(self, x, type):
        return x.to(type)

    @classmethod
    def to_numpy(self, x):
        return x

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


class CPUBasicDeviceOperations(TMBasicDeviceOperations):

    @classmethod
    def is_device(self, x):
        raise NotImplementedError()

    @classmethod
    def set_device(cls, device):
        pass

    @classmethod
    def cuda_available(cls):
        return False

    @classmethod
    def device_prefix(cls):
        return "cpu"



class DistributedBatchSampler:
    """ `BatchSampler` wrapper that distributes across each batch multiple workers.

    Args:
        batch_sampler (np.utils.data.sampler.BatchSampler)
        num_replicas (int, optional): Number of processes participating in distributed training.
        rank (int, optional): Rank of the current process within num_replicas.

    Example:
        >>> from np.utils.data.sampler import BatchSampler
        >>> from np.utils.data.sampler import SequentialSampler
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
            distributed_sampler = np.utils.data.DistributedSampler(batch, **self.kwargs)
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


class TMCPUBackend(TMBackend):
    Name = "cpu"

    NoParallelStrategy = CPUNoDistributedStrategy
    DataParallelStrategy = CPUDataParallelStrategy
    DistributedDataParallelStrategy = CPUDistributedDataParallelStrategy

    OptimizerBehaviors = CPUOptimizerBehaviors

    DataLoader = None
    Dataset = None
    IterableDataset = None

    Sampler = None
    BatchSampler = None
    SubsetRandomSampler = None # np.utils.data.SubsetRandomSampler
    SequenceSampler = None # np.utils.data.SequentialSampler
    RandomSampler = None # np.utils.data.RandomSampler
    SequentialSampler = None # np.utils.data.SequentialSampler
    DistributedSampler = None # np.utils.data.DistributedSampler
    DistributedBatchSampler = None # DistributedBatchSampler

    Types = CPUTypes
    BasicTensorOperations = CPUBasicTensorOperations
    BasicModuleOperations = None
    BasicDeviceOperations = CPUBasicDeviceOperations
    Dist = None
