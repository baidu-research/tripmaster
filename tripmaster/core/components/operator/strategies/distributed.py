
import abc
from typing import Type
import random
import random
import string


class TMDistributedStrategy(object):

    Name: str = None

    def __init__(self, operator, world_size, use_gpu):

        self.operator = operator
        self.use_gpu = use_gpu
        self.world_size = world_size

    @abc.abstractmethod
    def run(self, func, train_data_streams, runtime_options):
        pass

    @abc.abstractmethod
    def init(self, local_rank):
        pass

    @abc.abstractmethod
    def sync_loss(self, loss):
        pass


class TMDistributedStrategyFactory(object):

    instance = None

    def __init__(self):

        self.name_strategy_map = dict()

    def register(self, strategy: Type[TMDistributedStrategy]):

        self.name_strategy_map[strategy.Name] = strategy

    def strategies(self):

        for name in self.name_strategy_map.keys():
            yield name

    def has_strategy(self, name):
        return name in self.name_strategy_map

    def choose(self, name) -> TMDistributedStrategy:

        return self.name_strategy_map[name]

    @classmethod
    def get(cls):
        """

        Returns:

        """
        if cls.instance is None:
            cls.instance = TMDistributedStrategyFactory()

        return cls.instance
