from typing import Type

from tripmaster.core.concepts.component import TMConfigurable


class TMMetricLoggingStrategy(TMConfigurable):
    """
    TMMetricLoggingStrategy
    """
    Name: str = None

    def log(self, evaluation_results):
        """

        Args:
            epoch:
            step:
            metric:

        Returns:

        """

        pass


class TMMetricLoggingStrategyFactory(object):
    """
    TMMetricLoggingStrategyFactory
    """

    instance = None

    def __init__(self):

        self.name_strategy_map = dict()

    def register(self, strategy: Type[TMMetricLoggingStrategy]):

        self.name_strategy_map[strategy.Name] = strategy

    def strategies(self):

        for name in self.name_strategy_map.keys():
            yield name

    def choose(self, name) -> TMMetricLoggingStrategy:
        return self.name_strategy_map[name]

    @classmethod
    def get(cls):
        """

        Returns:

        """
        if cls.instance is None:
            cls.instance = TMMetricLoggingStrategyFactory()

        return cls.instance