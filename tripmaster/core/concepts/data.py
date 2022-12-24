"""
base class for data
"""
import abc
import enum
from collections import defaultdict
from typing import Dict, Type

from tripmaster import logging
from tripmaster.core.concepts.component import TMConfigurable, TMSerializable

logger = logging.getLogger(__name__)


class TMDataLevel(object):

    Task = "task"
    Problem = "problem"
    Machine = "machine"
    Memory = "memory"
    Batch = "batch"

    __model = {Task: Problem, Problem: Machine, Machine: Memory, Memory: Batch}
    __reconstruct = {Problem: Task, Machine: Problem, Memory: Machine, Batch: Memory}

    __upper_level = {
        Task: {},
        Problem: {Task},
        Machine: {Problem, Task},
        Memory: {Machine, Problem, Task},
        Batch: {Memory, Machine, Problem, Task}
    }

    __lower_level = {
        Task: {Problem, Machine, Memory, Batch},
        Problem: {Machine, Memory, Batch},
        Machine: {Memory, Batch},
        Memory: {Batch},
        Batch: {}
    }

    @classmethod
    def model(cls, level):
        return cls.__model[level]

    @classmethod
    def reconstruct(cls, level):
        return cls.__reconstruct[level]

    @classmethod
    def upper_level(cls, level):
        return cls.__upper_level[level]

    @classmethod
    def lower_level(cls, level):
        return cls.__lower_level[level]

    @classmethod
    def uri_key(cls, level):
        return f"^_^uri@{level}"



class TMDataChannel(TMConfigurable):
    """
    TMLearningDataContainer
    """

    def __init__(self, hyper_params=None, data=None, level: TMDataLevel=None):  # kwargs is for multiple inheritance
        super().__init__(hyper_params)
        self._data = data
        self._sample_num = None
        self.__level = level

    def support_random_batch(self):
        try:
            if len(self._data) is not None and hasattr(self._data, "__getitem__"):
                return True
            else:
                return False
        except:
            return False

    @property
    def sample_num(self):
        return self._sample_num

    @sample_num.setter
    def sample_num(self, num):

        assert num is not None and self._sample_num is None
        self._sample_num = num
        self._data = list(x for i, x in enumerate(self._data) if i < self.sample_num)


    @property
    def level(self):
        return self.__level

    @level.setter
    def level(self, level):
        self.__level = level

    @property
    def data(self):
        """

        Returns:

        """
        return self._data

    def __iter__(self):

        for x in self.data:
            yield x

    def __getitem__(self, item):

        return self.data[item]

    def __len__(self):
        return len(self._data)

    def degenerate(self):
        
        import types
        if isinstance(self._data, (list, tuple)):
            return
        else: # if isinstance(self._data, types.GeneratorType):
            self._data = list(self._data)



class TMDataStream(TMSerializable):
    """
    TMDataStream
    Some thoughts, but not feasible: "Note: as a fundamental components of TM which across all the data pipeline,
          it should not be subclassed and change its default behavior.
          For old code, please load the data in TMOfflineInputStream and then
          return a TMDataStream"
    """
    def __init__(self, hyper_params=None, level=None, states=None):

        super().__init__(hyper_params)

        self.__channels = dict()
        self.__level = level

        if states is not None:
            self.load_states(states)

    @property
    def level(self):
        return self.__level

    @level.setter
    def level(self, level):
        self.__level = level

    @property
    def channels(self):
        return self.__channels.keys()

    @property
    def learn_channels(self):
        return self.hyper_params.channels.learn if self.hyper_params.channels.learn else []

    def add_sampled_training_eval_channels(self):

        if not self.hyper_params.train_sample_ratio_for_eval or self.hyper_params.train_sample_ratio_for_eval <= 0:
            return 

        import random, copy 
        sampled_channels = []
        for channel in self.learn_channels:
            assert channel in self.__channels

            sampled = [copy.deepcopy(sample) for sample in self.__channels[channel]
                            if random.random() < self.hyper_params.train_sample_ratio_for_eval]
            if len(sampled) <= 0:
                continue
            sampled_channels.append(f"{channel}#sampled")

            self.__channels[f"{channel}#sampled"] = TMDataChannel(data=sampled, level=self.__level)
                
        self.hyper_params.channels.eval = list(set(list(self.hyper_params.channels.eval) + sampled_channels))

    @learn_channels.setter
    def learn_channels(self, value):
        self.hyper_params.channels.learn = tuple(value)

    @property
    def eval_channels(self):
        return self.hyper_params.channels.eval if self.hyper_params.channels.eval else []

    @eval_channels.setter
    def eval_channels(self, value):
        self.hyper_params.channels.eval = tuple(value)

    @property
    def inference_channels(self):
        return self.hyper_params.channels.inference if self.hyper_params.channels.inference else []

    @inference_channels.setter
    def inference_channels(self, value):
        self.hyper_params.channels.inference = tuple(value)

    def __getitem__(self, item):

        return self.__channels[item]

    def __setitem__(self, key, value):
        if not isinstance(value, TMDataChannel):
            value = TMDataChannel(data=value, level=self.__level)

        self.__channels[key] = value

    def test(self, test_config):

        for k, v in self.__channels.items():
            v.sample_num = test_config.sample_num

    def states(self):
        for k, v in self.__channels.items():
            v.degenerate()

        return {"channels": {k: v._data for k, v in self.__channels.items() if not k.endswith("#sampled")},
                "level": self.__level}

    def load_states(self, states):
        self.__level = states["level"]
        self.__channels = {k: TMDataChannel(data=v, level=self.__level)
                           for k, v in states["channels"].items()}



class TMSharedDataStream(TMDataStream):
    """
    TMSharedDataStream
    """

    DataStreams: Dict[str, Type[TMDataStream]] = None
    AlignFields: str = None

    def __init__(self, hyper_params=None, level=None, states=None):
        super().__init__(hyper_params, level=level, states=states)

        if states is not None:
            self.load_states(states)
        else:
            levels = set()
            stream_dict = dict()
            for key, V in self.DataStreams.items():
                stream = V(self.hyper_params[key])
                levels.add(stream.level)

                stream_dict[key] = stream
            assert len(levels) == 1
            self.level = list(levels)[0]

            self.build(stream_dict)

    def build(self, stream_dict):
        """

        Args:
            stream_dict:

        Returns:

        """
        learn_channel_set = set(tuple(x.learn_channels) for x in stream_dict.values())
        eval_channel_set = set(tuple(x.eval_channels) for x in stream_dict.values())
        inference_channel_set = set(tuple(x.eval_channels) for x in stream_dict.values())

        assert len(learn_channel_set) == 1
        assert len(eval_channel_set) == 1
        assert len(inference_channel_set) == 1

        self.learn_channels = list(learn_channel_set)[0]
        self.eval_channels = list(eval_channel_set)[0]
        self.inference_channels = list(inference_channel_set)[0]

        for channel in self.learn_channels + self.eval_channels + self.inference_channels:
            samples = defaultdict(dict)
            for task, stream in stream_dict.items():
                for data in stream[channel]:
                    samples[data[self.AlignFields]][task] = data
            self[channel] = list(samples.values())



class TMMultiDataStream(TMSerializable):
    """
    Each data stream for one machine. The data in each stream is not aligned, that is, they may not be the
    feature or target for the same object
    """
    DataStreams: Dict[str, Type[TMDataStream]] = None

    def __init__(self, hyper_params=None, level=None, states=None):
        super().__init__(hyper_params)

        self.__streams = dict()
        self.__level = level

        if states is not None:
            self.load_states(states)
        elif self.DataStreams is not None:
            levels = set()
            for key, V in self.DataStreams.items():
                self.__streams[key] = V(self.hyper_params[key], level=level)
                levels.add(self.__streams[key].level)

            assert len(levels) == 1
            self.__level = list(levels)[0]

    @property
    def level(self):
        return self.__level

    @level.setter
    def level(self, level):
        self.__level = level
        for k, v in self.__streams.items():
            v.level = level

    def streams(self):
        """

        Returns:

        """
        yield from self.__streams.keys()

    def __getitem__(self, item):

        return self.__streams[item]

    def __setitem__(self, key, value):
        self.__streams[key] = value

    def test(self, test_config):

        for k, v in self.__streams.items():
            v.test(test_config)

    def states(self):

        return {"streams": {k: v.states() for k, v in self.__streams.items()},
                "level": self.__level}

    def load_states(self, states):
        self.__level = states["level"]
        for k, v in states["streams"].items():
            if self.DataStreams is not None:
                self.__streams[k] = self.DataStreams[k](self.hyper_params[k], states=states[k], level=self.__level)
            else:
                self.__streams[k] = TMDataStream(self.hyper_params[k], states=states[k], level=self.__level)


class TMMergedDataChannel(TMConfigurable):
    """
    TMMergedDataChannel
    """

    def __init__(self, channel_dict):
        super().__init__(hyper_params=None)
        self.__channel_dict = channel_dict

    @property
    def sample_num(self):
        return min([v.sample_num for v in self.__channel_dict.values()])

    def __getitem__(self, item):
        return dict((k, self.__channel_dict[k][item]) for k in self.__channel_dict.keys())

    def __iter__(self):
        for i in range(self.sample_num):
            yield self[i]



