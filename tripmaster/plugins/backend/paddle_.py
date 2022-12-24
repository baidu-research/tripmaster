import collections
from typing import List, Iterator, Sized, Sequence
import numpy as np

from tripmaster.core.components.backend import TMOptimizerBehaviors, TMBasicTensorOperations, TMTypes, TMBackend, \
    TMBasicModuleOperations, TMBasicDeviceOperations
from paddle.distributed import fleet
import paddle.distributed as dist
import logging

from tripmaster.core.components.operator.strategies.distributed import TMDistributedStrategy

logger = logging.getLogger(__name__)

import paddle


strategy = fleet.DistributedStrategy()
fleet.init(is_collective=True, strategy=strategy)

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


class PaddleNoDistributedStrategy(TMDistributedStrategy):
    """
    PaddleNoDistributedStrategy
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
            device = "gpu:" + str(local_rank)
        else:
            device = "cpu"
            
        paddle.set_device(device)

#        self.operator.machine.to(device)

    def sync_loss(self, loss):
        """
        sync_loss
        """
        return loss


class PaddleDDPMachine(paddle.DataParallel):

    def __init__(self, machine, *args, **kwargs):

        super().__init__(machine, *args, **kwargs)

    def __getattr__(self, name):

        try:
            return paddle.DataParallel.__getattr__(self, name)
        except AttributeError as e:
            if hasattr(self.module, name):
                return getattr(self.module, name)
            else:
                print(f"what happens {self.__dict__}")
                raise e


class PaddleDistributedDataParallelStrategy(TMDistributedStrategy):
    """
    PaddleDistributedDataParallelStrategy
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

        dist.spawn(func, args=(self, train_data_streams, runtime_options),
                   nprocs=self.world_size)

        global DISTRIBUTED
        DISTRIBUTED = True

    def init(self, local_rank):
        """
        init_train_loop
        """

        self.distributed = True

        if self.use_gpu:
            device = "gpu:" + str(local_rank)
        else:
            device = "cpu"
        dist.init_parallel_env()

        self.operator.machine = PaddleDDPMachine(self.operator.machine)
        self.operator.machine.to(device)


    def sync_loss(self, loss):
        """
        sync_loss
        """

        return loss

class PaddleFleetMachine(object):

    def __init__(self, machine):

        super().__init__()
        
        self.machine = fleet.distributed_model(machine)

    def __getattr__(self, name):

        if hasattr(self.machine, name):
            return object.__getattribute__(self.machine, name)
        elif hasattr(self.machine._layers, name):
            return getattr(self.machine._layers, name)
        else:
            logger.error(f"missing attribute {name}")
            raise AttributeError(f"missing attribute {name}")

def parallel_getattr(parallel_machine, name):
   

    try:
        return paddle.DataParallel.__getattr__(parallel_machine, name)
    except AttributeError as e:
        if hasattr(parallel_machine._layers, name):
            return getattr(parallel_machine._layers, name)
        else:
            logger.error(f"missing attribute {name}")
            raise e

def make_fleet_optimizer(optimizer):

    if isinstance(optimizer, dict):
        return dict((k, make_fleet_optimizer(v)) for k, v in optimizer.items())
    else:
        return fleet.distributed_optimizer(optimizer)
    

class PaddleFleetParallelStrategy(TMDistributedStrategy):
    """
    PaddleFleetParallelStrategy
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

        # import port_for
        # self.distributed_port = port_for.select_random()

        # func(fleet.local_rank(), self.operator, train_data_streams, runtime_options)
        # print( "rank: ", dist.get_rank() )

        # paddle.device.set_device("gpu:"+str( dist.get_rank() ))

        func(dist.get_rank(), self.operator, train_data_streams, runtime_options)


        global DISTRIBUTED
        DISTRIBUTED = True

    def init(self, local_rank):
        """
        init_train_loop
        """

        self.distributed = True

        if self.use_gpu:
            device = "gpu:" + str(local_rank)
        else:
            device = "cpu"


        import types 
        self.operator.machine = PaddleFleetMachine(self.operator.machine)

        self.operator.optimizer = make_fleet_optimizer(self.operator.optimizer)


    def sync_loss(self, loss):
        """
        sync_loss
        """

        return loss


class PaddleOptimizerBehaviors(TMOptimizerBehaviors):


    @classmethod
    def optimizer_state_dict(cls, optimizer):

        T = PaddleBasicTensorOperations
        state = optimizer.state_dict()
        state = {k: T.to_numpy(T.to_device(v, "cpu")) if isinstance(v, T.Tensor) else v
                     for k, v in state.items()}
        return state

    @classmethod
    def load_optimizer_state(cls, optimizer, state_dict):

        T = PaddleBasicTensorOperations
        M = PaddleBasicModuleOperations
        state = {k: T.to_tensor(v)
                 if isinstance(v, np.ndarray) else v
                 for k, v in state_dict.items()}
        M.load_state_dict(optimizer, state)

    @classmethod
    def set_train_mode(self, machine):
        machine.train()

    @classmethod
    def set_inference_mode(self, machine):
        machine.eval()

    no_grad = paddle.no_grad

    @classmethod
    def clear_grad(self, optimizer):

        if isinstance(optimizer, collections.Sequence):
            for optim in optimizer:
                optim.clear_grad()
        elif isinstance(optimizer, collections.Mapping):
            for name, optimizer in optimizer.items():
                self.clear_grad(optimizer)
        else:
            optimizer.clear_grad()

    @classmethod
    def backward_loss(self, loss):
        loss.backward()

    @classmethod
    def optimizer_step(self, optimizer):
        if isinstance(optimizer, collections.Sequence):
            for optim in optimizer:
                optim.step()
        elif isinstance(optimizer, collections.Mapping):
            for name, optim in optimizer.items():
                self.optimizer_step(optim)
        else:
            optimizer.step()

    @classmethod
    def lrscheduler_step(self, lr_scheduler):
        if isinstance(lr_scheduler, collections.Sequence):
            for scheduler in lr_scheduler:
                scheduler.step()
        elif isinstance(lr_scheduler, collections.Mapping):
            for name, scheduler in lr_scheduler.items():
                self.lrscheduler_step(scheduler)
        else:
            lr_scheduler.step()

    @classmethod
    def clip_gradient_norm(self, parameters, clip_norm_threshold):
        """
        do nothing since paddle do the clip in the optimizer 
        """

    @classmethod
    def create_optimization_components(self, parameters,
                                   optimizer_type, optimizer_params,
                                   lr_scheduler_type, lr_scheduler_params,
                                   gradient_clip_val):

        grad_clip = paddle.nn.ClipGradByGlobalNorm(gradient_clip_val) \
            if gradient_clip_val is not None and gradient_clip_val > 0 else None

        if "lr" in optimizer_params:
            lr_scheduler_params.learning_rate = optimizer_params.lr
            del optimizer_params["lr"]
        lr_scheduler = lr_scheduler_type(**lr_scheduler_params)
        optimizer = optimizer_type(parameters=parameters, grad_clip=grad_clip,
                                   learning_rate=lr_scheduler,
                                   **optimizer_params)

        return optimizer, lr_scheduler


class PaddleTypes(TMTypes):

    Bool = paddle.bool
    Int8 = paddle.int8
    Int16 = paddle.int16
    Int32 = paddle.int32
    Int64 = paddle.int64
    Float16 = paddle.float16
    Float32 = paddle.float32
    Float64 = paddle.float64
    Float = paddle.float32

from more_itertools import chunked

class PaddleBasicTensorOperations(TMBasicTensorOperations):

    Tensor = paddle.Tensor

    @classmethod
    def is_tensor(self, x):
        return isinstance(x, paddle.Tensor)

    @classmethod
    def pad(self, input, pad, value=0.0):

        pad = sum(reversed(list(chunked(pad, 2))), [])

        if len(pad) < 2 * input.ndim:
            pad = [0] * (2 * input.ndim - len(pad)) + pad

        return paddle.nn.functional.pad(input, pad, mode="constant", value=value)

    @classmethod
    def to_tensor(self, x, dtype=None):
        return paddle.to_tensor(x, dtype)

    @classmethod
    def stack(self, list, dim):
        return paddle.stack(list, axis=dim)

    @classmethod
    def cast(self, x, type):
        return paddle.cast(x, type)

    @classmethod
    def to_numpy(self, x):
        return x.numpy()

    @classmethod
    def to_device(self, x, device):
        assert self.is_tensor(x)
        
        if str(x.place) == f"Place({device})":
            return x
        else:
            return paddle.to_tensor(x, place=device)

    @classmethod
    def device(cls, x):
        return x.place

    @classmethod
    def shape(self, x):
        return x.shape

    
class BatchSampler(paddle.io.Sampler):
    r"""Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler or Iterable): Base sampler. Can be any iterable object
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, sampler: paddle.io.Sampler, batch_size: int, drop_last: bool) -> None:
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        super().__init__()
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last


    def __iter__(self) -> Iterator[List[int]]:
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        # Can only be called if self.sampler has __len__ implemented
        # We cannot enforce this condition, so we turn off typechecking for the
        # implementation below.
        # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
        if self.drop_last:
            return len(self.sampler) // self.batch_size  # type: ignore[arg-type]
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size  # type: ignore[arg-type]


class SequentialSampler(paddle.io.Sampler):
    r"""Samples elements sequentially, always in the same order.

    Args:
        data_source (Dataset): dataset to sample from
    """
    data_source: Sized

    def __init__(self, data_source: Sized) -> None:
        self.data_source = data_source

    def __iter__(self) -> Iterator[int]:
        return iter(range(len(self.data_source)))

    def __len__(self) -> int:
        return len(self.data_source)

class SubsetRandomSampler(paddle.io.Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
        generator (Generator): Generator used in sampling.
    """
    indices: Sequence[int]

    def __init__(self, indices: Sequence[int]) -> None:
        super().__init__()
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in paddle.randperm(len(self.indices)))

    def __len__(self):
        return len(self.indices)



class DistributedSampler(paddle.io.Sampler):
    """ Iterable wrapper that distributes data across multiple workers.

    Args:
        iterable (iterable)
        num_replicas (int, optional): Number of processes participating in distributed training.
        rank (int, optional): Rank of the current process within ``num_replicas``.

    Example:
        >>> list(DistributedSampler(range(10), num_replicas=2, rank=0))
        [0, 2, 4, 6, 8]
        >>> list(DistributedSampler(range(10), num_replicas=2, rank=1))
        [1, 3, 5, 7, 9]
    """

    def __init__(self, iterable, num_replicas=None, rank=None):
        self.iterable = iterable
        self.num_replicas = num_replicas
        self.rank = rank

        if num_replicas is None or rank is None:  # pragma: no cover
#                if not paddle.distributed.is_initialized():
#                    raise RuntimeError('Requires `torch.distributed` to be initialized.')

            self.num_replicas = (
                paddle.distributed.get_world_size() if num_replicas is None else num_replicas)
            self.rank = paddle.distributed.get_rank() if rank is None else rank

        if self.rank >= self.num_replicas:
            raise IndexError('`rank` must be smaller than the `num_replicas`.')

    def __iter__(self):
        return iter(
            [e for i, e in enumerate(self.iterable) if (i - self.rank) % self.num_replicas == 0])

    def __len__(self):
        return len(self.iterable)


class PaddleBasicDeviceOperations(TMBasicDeviceOperations):

    @classmethod
    def is_device(self, x):
        raise NotImplementedError()

    @classmethod
    def set_device(cls, device):
        paddle.set_device(device)

    @classmethod
    def cuda_available(cls):
        raise NotImplementedError()
        
    @classmethod
    def device_prefix(cls):
        return "gpu"



class PaddleBasicModuleOperations(TMBasicModuleOperations):

    Module = paddle.nn.Layer
    ModuleList = paddle.nn.LayerList
    ModuleDict = paddle.nn.LayerDict

    @classmethod
    def load_state_dict(cls, module: paddle.nn.Layer, state_dict):
        module.set_state_dict(state_dict)

    @classmethod
    def to_device(cls, module, device):
        return module

class TMPaddleBackend(TMBackend):

    Name = "paddle"

    NoParallelStrategy = PaddleNoDistributedStrategy
    DataParallelStrategy = None
    DistributedDataParallelStrategy = PaddleFleetParallelStrategy

    OptimizationModule = paddle.optimizer
    OptimizerBehaviors = PaddleOptimizerBehaviors

    DataLoader = paddle.io.DataLoader
    Dataset = paddle.io.Dataset
    IterableDataset = paddle.io.IterableDataset

    Sampler = paddle.io.Sampler
    BatchSampler = BatchSampler
    SubsetRandomSampler = SubsetRandomSampler
    SequenceSampler = paddle.io.SequenceSampler
    RandomSampler = paddle.io.RandomSampler
    SequentialSampler = SequenceSampler
    DistributedSampler = DistributedSampler
    DistributedBatchSampler = paddle.io.DistributedBatchSampler

    Types = PaddleTypes
    BasicTensorOperations = PaddleBasicTensorOperations
    BasicModuleOperations = PaddleBasicModuleOperations
    BasicDeviceOperations = PaddleBasicDeviceOperations
    Dist = paddle.distributed
