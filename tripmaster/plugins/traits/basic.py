import collections
from collections.abc import Sequence
from more_itertools import ichunked

from tripmaster.core.components.machine.data_traits import TMElementTraits, TMElementBatchTraits, \
    TMElementTraitsFactory, UnsupportedTraitsError
import numpy as np

from tripmaster.core.components.backend import TMBackendFactory

B = TMBackendFactory.get().chosen()
T = B.BasicTensorOperations

class TMIntElementTraits(TMElementTraits):

    ElementType = (bool, int, np.integer)

    @classmethod
    def collate(self, list_of_samples):

        target_type = B.Types.Bool if isinstance(list_of_samples[0], bool) else B.Types.Int64
        return T.cast(T.to_tensor(np.array(list_of_samples)), target_type)


class TMFloatElementTraits(TMElementTraits):

    ElementType = (float, np.float64, np.float32)

    @classmethod
    def collate(self, list_of_samples):
        return T.to_tensor(np.array(list_of_samples)).cast(B.Types.Float)


class TMTensorElementTraits(TMElementTraits):

    ElementType = (np.ndarray, T.is_tensor)

    @classmethod
    def collate(self, list_of_samples):

        first_elem = list_of_samples[0]
        # if isinstance(first_elem, np.ndarray):
        #     list_of_samples = [T.to_tensor(b) for b in list_of_samples]
        
        shapes = [list(x.shape if isinstance(x, np.ndarray) else T.shape(x)) for x in list_of_samples]

        max_shape = tuple([max(x[i] for x in shapes) for i in range(first_elem.ndim)])
    

        for idx, b in enumerate(list_of_samples):
            this_shape = b.shape if isinstance(b, np.ndarray) else T.shape(b)
            if tuple(this_shape) == max_shape:
                continue 
                
            if isinstance(first_elem, np.ndarray):
                pad = [(0, max_shape[i] - this_shape[i]) for i in range(b.ndim)]
                b = np.pad(b, pad)
            else:  # is tensor 

                dim_order = range(first_elem.ndim - 1, -1, -1)

                pad = [[0, max_shape[i] - this_shape[i]] for i in dim_order]
                pad = sum(pad, [])
                b = T.pad(b, pad, 0)

            list_of_samples[idx] = b
        
        if isinstance(first_elem, np.ndarray):
            result = T.to_tensor(np.stack(list_of_samples, 0))
        else:
            result = T.stack(list_of_samples, 0)
        return result


class TMTensorElementBatchTraits(TMElementBatchTraits):

    ElementBatchType = T.is_tensor

    @classmethod
    def decollate(self, batch):

        batch_size = T.shape(batch)[0]
        batch = T.to_numpy(T.to_device(batch, device="cpu"))
        return [batch[i] for i in range(batch_size)]

    @classmethod
    def batch_size(cls, batch):

        return T.shape(batch)[0]

    @classmethod
    def to_device(cls, batch, device):
        return T.to_device(batch, device)

    @classmethod
    def shape(cls, batch):
        return T.shape(batch)



class TMDictElementTraits(TMElementTraits):
    """
    TMDictElementBatchTraits
    """
    ElementType = (dict, )

    @classmethod
    def collate(self, samples):
        """
        collate
        """
        assert isinstance(samples, Sequence)

        first_elem = samples[0]
        assert isinstance(first_elem, dict)

        batched = dict()

        for key in first_elem.keys():

            values = [d[key] for d in samples]
            try:
                traits = TMElementTraitsFactory.get().get_element_traits(values)
                batched[key] = traits.collate(values)
            except UnsupportedTraitsError as e:
                batched[key] = values

        return batched
                    # raise e

class TMDictElementBatchTraits(TMElementBatchTraits):
    """
    TMDictElementBatchTraits
    """
    ElementBatchType = (dict, )

    @classmethod
    def decollate(cls, batched_data):
        """
        decollate
        """
        assert isinstance(batched_data, dict)

        batch_size = cls.batch_size(batched_data)

        unbatched_data = collections.defaultdict(list)

        for key in batched_data.keys():
            try:
                traits = TMElementTraitsFactory.get().get_element_batch_traits(batched_data[key])
                unbatched_data[key] = traits.decollate(batched_data[key])
            except UnsupportedTraitsError as e:
                unbatched_data[key] = batched_data[key]

        result_list = list()
        key_list = list(batched_data.keys())

        for sample in zip(*(unbatched_data[key] for key in key_list)):
            result_list.append(dict(zip(key_list, sample)))

        return result_list

    @classmethod
    def batch_size(cls, batch):
        """
        batch_size
        """
        for key in batch.keys():
            try:
                traits = TMElementTraitsFactory.get().get_element_batch_traits(batch[key])
                return traits.batch_size(batch[key])
            except UnsupportedTraitsError as e:
                continue

        raise UnsupportedTraitsError("Unable to obtain batch size ")

    @classmethod
    def to_device(cls, batch, device):
        """
        to_device
        """
        result = dict()
        for key in batch.keys():
            try:
                traits = TMElementTraitsFactory.get().get_element_batch_traits(batch[key])
                result[key] = traits.to_device(batch[key], device)
            except UnsupportedTraitsError as e:
                result[key] = batch[key]
        return result


    @classmethod
    def shape(cls, batch):
        """
        shape
        """
        shape_dict = dict()

        for key in batch.keys():

            try:
                traits = TMElementTraitsFactory.get().get_element_batch_traits(batch[key])
                shape_dict[key] = traits.shape(batch[key])
            except UnsupportedTraitsError as e:
                pass

        return shape_dict


class TMListElementBatchTraits(TMElementBatchTraits):
    """
    TMDictElementBatchTraits
    """
    ElementBatchType = (tuple, list)

    @classmethod
    def decollate(cls, batched_data):
        """
        decollate
        """
        assert isinstance(batched_data, list)
        return batched_data

    @classmethod
    def batch_size(cls, batch):
        """
        batch_size
        """
        return len(batch)

    @classmethod
    def to_device(cls, batch, device):
        """
        to_device
        """

        return batch


    @classmethod
    def shape(cls, batch):
        """
        shape
        """

        return len(batch)
