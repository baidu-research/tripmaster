"""
resource
"""

import collections
import abc
from more_itertools import ichunked
import collections
import inspect
import types
from typing import Type
from collections.abc import Sequence, Mapping


class UnsupportedTraitsError(Exception):

    def __init__(self, message):
        super().__init__(message)


class TMElementTraits(object):
    ElementType = None

    @classmethod
    def collate(self, list_of_sample):
        pass


class TMElementBatchTraits(object):
    ElementBatchType = None

    @classmethod
    def decollate(self, batch):
        pass

    @classmethod
    def to_device(cls, batch, device):
        pass

    @classmethod
    def batch_size(cls, batch):
        pass

    @classmethod
    def shape(cls, batch):
        pass


class TMElementTraitsFactory(object):
    instance = None

    def __init__(self):

        self.element_traits_map = dict()
        self.batch_traits_map = dict()

    def register_element_strategy(self, strategy: Type[TMElementTraits]):

        sample_types = strategy.ElementType if isinstance(strategy.ElementType, Sequence) \
            else [strategy.ElementType]
        for t in filter(None, sample_types):
                self.element_traits_map[t] = strategy

    def register_element_batch_strategy(self, strategy: Type[TMElementBatchTraits]):

        batch_types = strategy.ElementBatchType if isinstance(strategy.ElementBatchType, Sequence) \
            else [strategy.ElementBatchType]
        for t in filter(None, batch_types):
            self.batch_traits_map[t] = strategy

    def get_element_traits(self, list_of_elems) -> TMElementTraits:
        element = list_of_elems[0]
        elem_type = type(element)

        if elem_type in self.element_traits_map:
            return self.element_traits_map[elem_type]
        else:
            for key in self.element_traits_map.keys():
                if isinstance(key, (types.FunctionType, types.MethodType)) and key(element):
                    return self.element_traits_map[key]
    #
            # for t in inspect.getmro(elem_type):
            #     if t in self.element_traits_map:
            #         return self.element_traits_map[t]
        raise UnsupportedTraitsError(f"Unsupported element type {elem_type}")

    def get_element_batch_traits(self, batch) -> TMElementBatchTraits:

        batch_type = type(batch)

        if batch_type in self.batch_traits_map:
            return self.batch_traits_map[batch_type]
        else:
            for key in self.batch_traits_map.keys():
                if isinstance(key, (types.FunctionType, types.MethodType)) and key(batch):
                    return self.batch_traits_map[key]

        raise UnsupportedTraitsError(f"Unsupported batch type {batch_type}")

    @classmethod
    def get(cls):
        """

        Returns:


        """
        if cls.instance is None:
            cls.instance = TMElementTraitsFactory()

        return cls.instance


class TMSampleMemoryTraits(abc.ABC):
    SAMPLE_OOM_POSSIBLE = False

    @classmethod
    @abc.abstractmethod
    def padding_dim(self):

        raise NotImplementedError()

    @classmethod
    @abc.abstractmethod
    def memory_request(self, data) -> dict:
        """

        Args:
            data:

        Returns:

        """
        raise NotImplementedError()

    @classmethod
    @abc.abstractmethod
    def fit_memory(self, data, resource_limit):
        """

        Args:
            data:

        Returns:

        """
        yield data

    @classmethod
    @abc.abstractmethod
    def unfit_memory(cls, list_of_data):
        """

        Args:
            list_of_data:

        Returns:

        """
        if len(list_of_data) == 1:
            return list_of_data[0]
        else:
            raise NotImplementedError()


class TMSampleBatchTraits(object):

    # @classmethod
    # def duplicate(cls, batch, times):
    #     for key, v in batch.items():
    #
    #         if isinstance(v, B.Tensor):
    #             repeat_dim = [1] * v.ndim
    #             repeat_dim[0] = times
    #             batch[key] = v.repeat(*repeat_dim)

    @classmethod
    def batch(cls, samples):

        assert isinstance(samples, Sequence)
        traits = TMElementTraitsFactory.get().get_element_traits(samples)
        return traits.collate(samples)
        #
        #
        # assert isinstance(first_elem, Mapping)
        #
        # batched = dict()
        # for key in first_elem.keys():
        #     values = [d[key] for d in samples]
        #
        #     try:
        #
        #         traits = TMElementTraitsFactory.get().get_element_traits(values)
        #
        #         batched[key] = traits.collate(values)
        #
        #     except UnsupportedTraitsError as e:
        #         batched[key] = values
        #
        # return batched

    
    @classmethod
    def unbatch(cls, batched_data):
        """
            decollate a collated batch into a list of samples
            Args:
                batch ():

            Returns:

            """

        traits = TMElementTraitsFactory.get().get_element_batch_traits(batched_data)
        return traits.decollate(batched_data)

        # assert isinstance(batched_data, Mapping)
        #
        # batch_size = cls.batch_size(batched_data)
        #
        # unbatched_data = collections.defaultdict(list)
        #
        # for key in batched_data.keys():
        #     try:
        #         traits = TMElementTraitsFactory.get().get_element_batch_traits(batched_data[key])
        #         unbatched_data[key] = traits.decollate(batched_data[key])
        #     except UnsupportedTraitsError as e:
        #         unbatched_data[key] = batched_data[key]
        #
        # return [dict((key, unbatched_data[key][i]) for key in batched_data.keys()) for i in range(batch_size)]

    @classmethod
    def batch_size(cls, batched_data):
        """

        @param data:
        @type data:
        @return:
        @rtype:
        """
        traits = TMElementTraitsFactory.get().get_element_batch_traits(batched_data)
        return traits.batch_size(batched_data)
        # for key in batched_data.keys():
        #     try:
        #
        #     except UnsupportedTraitsError as e:
        #         pass
        #
        # raise Exception("cannot calculate batch size")


    @classmethod
    def shape(cls, batched_data):
        """

        @param data:
        @type data:
        @return:
        @rtype:
        """
        traits = TMElementTraitsFactory.get().get_element_batch_traits(batched_data)
        data_shape = traits.shape(batched_data)

        return data_shape

    @classmethod
    def to_device(cls, batched_data, device):
        """

        @param data:
        @type data:
        @return:
        @rtype:
        """

        traits = TMElementTraitsFactory.get().get_element_batch_traits(batched_data)
        target_data = traits.to_device(batched_data, device)

        return target_data


class TMSampleChunckBatchTraits(TMSampleBatchTraits):

    @classmethod
    def batch(cls, samples):
        # print("len samples in batch traits: ", len(samples))
        # print("sampl")
        
        assert isinstance(samples, Sequence)

        first_elem = samples[0]
        # print("fe l:", len(first_elem))
        # print("first f e: ", first_elem[0].keys())
        # exit(0)
        assert isinstance(first_elem, Mapping)

        batched = dict()
        
        for key in first_elem.keys():
            # print("key: ", key)
            # for d in samples:
                # print("d key: ", d.keys())
            values = [torch.tensor(d[key], dtype=torch.float32) for d in samples]
            # print("len values: ", len(values))
            # print("values shape: ", type(values[0]))
            try:
                v = torch.cat(values, dim=0)
            except:
                # print("except key: ", key)
                v = v
            # exit(0)
            # print("v", v.shape)
            batched[key] = v
            # try:
            #     traits = TMElementTraitsFactory.get().get_element_traits(values)
            #     batched[key] = traits.collate(values)
            # except UnsupportedTraitsError as e:
            #     batched[key] = values
        # print("batched: ", batched.keys())
        return batched

    @classmethod
    def unbatch(cls, batched_data):
        """
            decollate a collated batch into a list of samples
            Args:
                batch ():

            Returns:

            """
        assert isinstance(batched_data, Mapping)

        batch_size = cls.batch_size(batched_data)

        unbatched_data = collections.defaultdict(list)

        for key in batched_data.keys():
            try:
                traits = TMElementTraitsFactory.get().get_element_batch_traits(batched_data[key])
                unbatched_data[key] = traits.decollate(batched_data[key])
            except UnsupportedTraitsError as e:
                unbatched_data[key] = batched_data[key]

        return [dict((key, unbatched_data[key][i]) for key in batched_data.keys()) for i in range(batch_size)]


class TMMultiTaskSampleBatchTraits(object):

    @classmethod
    def batch(cls, samples):
        """
        collate list of samples into a batches
        """

        assert isinstance(samples, Sequence)

        first_elem = samples[0]
        assert isinstance(first_elem, Mapping)

        batched = dict()

        def batch_dictdata(samples):
            """
            recursively batch dictionaries of data
            """

            first_elem = samples[0]

            if isinstance(first_elem, Mapping):
                data_dict = {}
                for key in first_elem.keys():
                    values = [d[key] for d in samples]
                    data_dict[key] = batch_dictdata(values)
                return data_dict
            else:
                try:
                    traits = TMElementTraitsFactory.get().get_element_traits(samples)
                    return traits.collate(samples)
                except UnsupportedTraitsError as e:
                    return samples

        batched = batch_dictdata(samples)

        return batched

    @classmethod
    def unbatch(cls, batched_data):
        """
        decollate a collated batch into a list of samples
        Args:
            batch ():

        Returns:

        """

        assert isinstance(batched_data, Mapping)

        batch_size = cls.batch_size(batched_data)

        unbatched_data = collections.defaultdict(list)

        def unbatch_dictdata(data):
            """
            recursively unbatch dictionaries of data
            """
            if isinstance(data, Mapping):
                data_dict = {}
                for key in data.keys():
                    data_dict[key] = unbatch_dictdata(data[key])
                return data_dict
            else:
                try:
                    traits = TMElementTraitsFactory.get().get_element_batch_traits(data)
                    return traits.decollate(data)
                except UnsupportedTraitsError as e:
                    return data

        unbatched_data = unbatch_dictdata(batched_data)

        def itemize_data(data, i):
            """
            itemize unbatched data
            """

            if isinstance(data, Mapping):
                data_dict = {}
                for key in data.keys():
                    data_dict[key] = itemize_data(data[key], i)
                return data_dict
            else:
                if isinstance(data, list):
                    try:
                        return data[i]
                    except TypeError as e:
                        pass
                else:
                    return data

        result_list = list()
        for i in range(batch_size):
            try:
                result = itemize_data(unbatched_data, i)
            except ValueError as e:
                pass

            result_list.append(itemize_data(unbatched_data, i))

        return result_list

    @classmethod
    def batch_size(cls, batched_data):
        """

        @param data:
        @type data:
        @return:
        @rtype:
        """

        for key in batched_data.keys():
            if isinstance(batched_data[key], Mapping):
                return cls.batch_size(batched_data[key])
            else:
                try:
                    traits = TMElementTraitsFactory.get().get_element_batch_traits(batched_data[key])
                    return traits.batch_size(batched_data[key])
                except UnsupportedTraitsError as e:
                    pass

        raise Exception("cannot calculate bath size")

    @classmethod
    def shape(cls, batched_data):
        """

        @param data:
        @type data:
        @return:
        @rtype:
        """

        if isinstance(batched_data, Mapping):
            shape_dict = {}
            for key in batched_data.keys():
                shape_dict[key] = cls.shape(batched_data[key])
            return shape_dict
        else:
            try:
                traits = TMElementTraitsFactory.get().get_element_batch_traits(batched_data)
                return traits.shape(batched_data)
            except UnsupportedTraitsError as e:
                return len(batched_data)

    @classmethod
    def to_device(cls, batched_data, device):
        """

        @param data:
        @type data:
        @return:
        @rtype:
        """

        if isinstance(batched_data, Mapping):
            target_data = {}
            for key in batched_data.keys():
                target_data[key] = cls.to_device(batched_data[key], device)
            return target_data
        else:
            try:
                traits = TMElementTraitsFactory.get().get_element_batch_traits(batched_data)
                return traits.to_device(batched_data, device)
            except UnsupportedTraitsError as e:
                return batched_data

