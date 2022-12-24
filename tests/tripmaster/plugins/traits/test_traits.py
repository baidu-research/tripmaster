import dataclasses

from icecream import install
install()

from tripmaster.core.components.machine.data_traits import TMSampleBatchTraits, TMElementTraitsFactory
from tripmaster import P, T

import numpy as np

def test_traits_factory():

    factory = TMElementTraitsFactory.get()

    assert int in factory.element_traits_map
    assert np.ndarray in factory.element_traits_map
    assert T.is_tensor in factory.element_traits_map
    assert dataclasses.is_dataclass in factory.element_traits_map
    assert T.is_tensor in factory.batch_traits_map
    assert dataclasses.is_dataclass in factory.batch_traits_map


def test_lists_traits():

    data = [{"x": np.zeros((20,))}] * 10

    batch = TMSampleBatchTraits.batch(data)

    assert isinstance(batch, dict) and "x" in batch
    assert P.BasicTensorOperations.is_tensor(batch["x"])
    assert tuple(P.BasicTensorOperations.shape(batch["x"])) == (10, 20)


