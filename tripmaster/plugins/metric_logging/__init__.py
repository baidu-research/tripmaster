"""
metric logging
"""


from tripmaster.core.components.operator.strategies.metric_logging import TMMetricLoggingStrategy, \
    TMMetricLoggingStrategyFactory


from tripmaster.plugins.load import load_plugins

register_funcs = {TMMetricLoggingStrategy: TMMetricLoggingStrategyFactory.get().register}

load_plugins(__name__, __file__, register_funcs)
