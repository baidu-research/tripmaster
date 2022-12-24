


from tripmaster.core.components.operator.strategies.batching import TMBatchingStrategy, TMBatchingStrategyFactory

from tripmaster.plugins.load import load_plugins

register_funcs = {TMBatchingStrategy: TMBatchingStrategyFactory.get().register}

load_plugins(__name__, __file__, register_funcs)
