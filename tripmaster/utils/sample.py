"""
sample
"""
import collections

import numpy as np
from tripmaster import T

def batchwise_select(batch_sample, index):
    """
    sample_index
    """

    if isinstance(batch_sample, np.ndarray) or T.is_tensor(batch_sample):

        result = batch_sample[np.arange(batch_sample.shape[0]), index]
        return result
    elif isinstance(batch_sample, collections.Sequence):
        return [batchwise_select(x, index) for x in batch_sample]
    elif isinstance(batch_sample, collections.Mapping):
        return dict((k, batchwise_select(v, index)) for k, v in batch_sample.items())
    else:
        raise TypeError("sample type not supported: {}".format(type(batch_sample)))
