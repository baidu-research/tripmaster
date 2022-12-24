"""
evaluator
"""
import copy
from abc import ABC, abstractmethod
from enum import auto
from typing import Dict, Any

from tripmaster.core.concepts.contract import TMContracted


class TMEvaluatorInterface(TMContracted):


    @abstractmethod
    def update(self, machine, truth):  # pylint: disable=E0202
        """
        Iteratively call update for each metric. Positional arguments (args) will
        be passed to every metric in the collection, while keyword arguments (kwargs)
        will be filtered based on the signature of the individual metric.
        """

        raise NotImplementedError()

    @abstractmethod
    def compute(self) -> Dict[str, Any]:
        """

        Returns:

        """
        raise NotImplementedError()

    @abstractmethod
    def metric_names(self):
        raise NotImplementedError()

    @abstractmethod
    def reset(self):
        raise NotImplementedError()

    def clone(self):
        return copy.deepcopy(self)

    def to(self, device):
        pass

    def set_world_size(self, world_size):
        pass


            #
    # def forward(self, machine, truth) -> Dict[str, Any]:  # pylint: disable=E0202
    #     """
    #     Iteratively call forward for each metric. Positional arguments (args) will
    #     be passed to every metric in the collection, while keyword arguments (kwargs)
    #     will be filtered based on the signature of the individual metric.
    #     """
    #
    #     pred_args = [machine[x] for x in self.requires("machine")]
    #     truth_args = [truth[x] for x in self.requires(("truth"))]
    #     arg_list = pred_args + truth_args
    #
    #     result = dict()
    #     for k, m in self.items():
    #         ret = m(*arg_list)
    #         if isinstance(ret, T.Tensor):
    #             result[k] = ret
    #         elif isinstance(ret, dict):
    #             for ret_k, v in ret.items():
    #                 result[".".join((k, ret_k))] = v
    #         else:
    #             raise Exception(f"Unknown type of metric returns {type(ret)}")
    #
    #     return result


import abc


class TMEvaluationStrategy(object):
    """
    EvaluationCallback
    """

    def __init__(self, evaluator: TMEvaluatorInterface):
        """

        Args:
            problem_evaluator:
            problem_modeler:
            task_evaluator:
        """

        self.evaluator_prototype = evaluator
        self.channel_evaluator = dict()

        self.channel_performance = dict()

    def get_evaluator(self, channel, device):

        #        self.task_evaluator = task_evaluator
        #        self.modeler = modeler

        if channel not in self.channel_evaluator:

            evaluator = self.evaluator_prototype.clone()
            evaluator.to(device)
            # evaluator.set_world_size(world_size)
            # evaluator.set_unic_id(unic_id)
            evaluator.reset()
            self.channel_evaluator[channel] = evaluator

        return self.channel_evaluator[channel]

    def on_evaluation_begin(self):

        for channel, evaluator in self.channel_evaluator.items():
            evaluator.reset()

        self.channel_performance = dict()

    @abc.abstractmethod
    def on_stream_inferenced(self, info):
        """
        local_rank, epoch = info
        Args:
            inference_stream:
            scenario:
            local_rank:

        Returns:

        """

        pass

    @abc.abstractmethod
    def on_evaluation_end(self, info):
        """
        local_rank, epoch = info
        Args:
            inference_stream:
            scenario:
            local_rank:

        Returns:

        """

        pass

