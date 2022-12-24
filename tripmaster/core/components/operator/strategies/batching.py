from typing import Type

from tripmaster.core.concepts.component import TMConfigurable


class TMBatchingStrategy(TMConfigurable):

    Name: str = None

    def batch_sampler(self, data_channel, data_traits, learning, distributed=False):

        pass



class TMBatchingStrategyFactory(object):

    instance = None

    def __init__(self):

        self.name_strategy_map = dict()

    def register(self, strategy: Type[TMBatchingStrategy]):

        self.name_strategy_map[strategy.Name] = strategy

    def strategies(self):

        for name in self.name_strategy_map.keys():
            yield name

    def choose(self, name) -> TMBatchingStrategy:

        return self.name_strategy_map[name]

    @classmethod
    def get(cls):
        """

        Returns:

        """
        if cls.instance is None:
            cls.instance = TMBatchingStrategyFactory()

        return cls.instance