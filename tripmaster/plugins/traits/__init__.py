

from tripmaster.core.components.machine.data_traits import TMElementTraits, TMElementBatchTraits, TMElementTraitsFactory

from tripmaster.plugins.load import load_plugins

register_funcs = {TMElementTraits: TMElementTraitsFactory.get().register_element_strategy,
                  TMElementBatchTraits: TMElementTraitsFactory.get().register_element_batch_strategy}

load_plugins(__name__, __file__, register_funcs)
