

from tripmaster.core.components.backend import TMBackendFactory
from tripmaster.core.components.operator.strategies.distributed import TMDistributedStrategyFactory

B = TMBackendFactory.get().chosen()

if B.NoParallelStrategy:
    TMDistributedStrategyFactory.get().register(B.NoParallelStrategy)

if B.DataParallelStrategy:
    TMDistributedStrategyFactory.get().register(B.DataParallelStrategy)

if B.DistributedDataParallelStrategy:
    TMDistributedStrategyFactory.get().register(B.DistributedDataParallelStrategy)


def get_distributed_settings():
    if B.Name == "torch":
        if B.Dist.is_available() and B.Dist.is_initialized():
            return B.Dist.get_world_size(), B.Dist.get_rank()
        else:
            return 1, 0
    elif B.Name == "paddle":
        return B.Dist.get_world_size(), B.Dist.get_rank()
    else:
        raise Exception("Unknown P.Name: {}".format(B.Name))