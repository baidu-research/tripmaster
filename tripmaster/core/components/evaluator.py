import copy
import inspect
import itertools

from typing import Dict, Any, Union, List

import more_itertools

from tripmaster import logging

from tripmaster.core.concepts.component import TMSerializableComponent, TMComponent
from tripmaster.core.concepts.contract import TMContract, TMContractChannel, TMContracted
from tripmaster.core.components.contract import TMContractGraph
from tripmaster.core.concepts.evaluator import TMEvaluatorInterface

from tripmaster.core.components.backend import TMBackendFactory
from tripmaster.core.concepts.schema import TMSchema, TMChannelSchema

B = TMBackendFactory.get().chosen()
M = B.BasicModuleOperations
T = B.BasicTensorOperations

logger = logging.getLogger(__name__)



class TMEvaluator(TMSerializableComponent, TMEvaluatorInterface):
    """
    TMEvaluator
    """
    pass

class TMMetricEvaluator(TMEvaluator):
    """
    TMMetricEvaluator: Define an evaluator from existing metric
    The evaluator support multiple-metrics with unified input schema.
    It also supports inline schema construction. So
       * if u need re-usable component, you explicitly subclass the evaluator, define the ForwardRequestSchema
         in the subclass, and then define the contract in the Task/Problem/Machine
       * if u want fast programming or one-time usage, you can initialize the evaluator by provide the initialization
         parameters, where the `inference_fields` and `truth_fields` will be used to create the schema
    """

    Metrics: Dict[str, Any] = None

    @classmethod
    def init_class(cls):
        requests = dict((x, object) for x in cls.Metrics)
        cls.ForwardRequestSchema = TMChannelSchema(
            {TMContractChannel.Truth: requests, TMContractChannel.Inference: requests})


    @classmethod
    def forward_request_schema(cls):

        requests = dict((x, object) for x in cls.Metrics)
        return TMChannelSchema(
            {TMContractChannel.Truth: requests, TMContractChannel.Inference: requests})


    def __init__(self, hyper_params=None, states=None):
        """

        Args:
            metrics: dict of (field, metric_lists). field denotes the data field for evaluation, metric list denotes the
                     multiple metrics to evaluate
        """
        super().__init__(hyper_params=hyper_params, states=states)
        self.metrics = copy.deepcopy(self.Metrics)

        #
        # if inference_fields:
        #     if TMContractChannel.Inference in self.ForwardRequestSchema:
        #         logger.warning("TMMetricEvaluator: the inference schema will "
        #                        "be overwritten by user provided info")
        #     self.ForwardRequestSchema[TMContractChannel.Inference] = \
        #         dict((x, T.Tensor) for x in inference_fields)
        # if truth_fields:
        #     if TMContractChannel.Truth in self.ForwardRequestSchema:
        #         logger.warning("TMMetricEvaluator: the truth schema will"
        #                        " be overwritten by user provided info")
        #     self.ForwardRequestSchema[TMContractChannel.Truth] = \
        #         dict((x, T.Tensor) for x in truth_fields)

    def update(self, machine, truth):  # pylint: disable=E0202
        """
        Iteratively call update for each metric. Positional arguments (args) will
        be passed to every metric in the collection, while keyword arguments (kwargs)
        will be filtered based on the signature of the individual metric.
        """

        pred_args = [machine[x] for x in self.forward_request_schema()[TMContractChannel.Inference].entries()]
        truth_args = [truth[x] for x in self.forward_request_schema()[TMContractChannel.Truth].entries()]
        arg_list = pred_args + truth_args
        for m in self.metrics.values():
            if isinstance(m, (list, tuple)):
                for x in m:
                    x.update(*arg_list)
            else:
                m.update(*arg_list)

    def __metric_compute(self, metric):

        if B.Name == "paddle":
            import paddle
            if isinstance(metric, paddle.metric.Metric):
                return {metric.name(): metric.accumulate()}
            else: # paddlemetric
                return metric.compute()
        else:
            return metric.compute()

    def compute(self) -> Dict[str, Any]:
        """

        Returns:

        """
        result = dict()
        for key, m in self.metrics.items():
            if isinstance(m, (list, tuple)):
                ret = [self.__metric_compute(x) for x in m]
            else:
                ret = [self.__metric_compute(m)]
            for idx, x in enumerate(ret):
                if isinstance(x, dict):
                    for ret_k, v in x.items():
                        result[f"{key}.{ret_k}"] = v
                else:
                    if len(ret) == 1:
                        result[key] = ret
                    else:
                        result[f"{key}.{idx}"] = ret
        return result

    def clone(self):

        import copy
        return copy.deepcopy(self)

    def reset(self):
        for key, metric in self.metrics.items():
            if isinstance(metric, (tuple, list)):
                for x in metric:
                    x.reset()
            else:
                metric.reset()

    def __metric_to(self, metric, device):

        if B.Name == "paddle":
            return metric
        else:
            if isinstance(metric, (tuple, list)):
                return [x.to(device) for x in metric]
            else:
                return metric.to(device)

    def to(self, device):
        for key, metric in self.metrics.items():
            self.metrics[key] = self.__metric_to(metric, device)


class TMEvaluatorCollection(M.ModuleDict, TMEvaluator):
    """
    TMEvaluatorCollection:
    multiple evaluator for same data
    """

    def __init__(self, modules: Dict[str, Any]):
        """

        Args:
            modules:
        """

        # donot call super().__init__(modules), because it calls the update function
        M.ModuleDict.__init__(self)
        TMEvaluator.__init__(self, hyper_params=None)
        self.non_metric_evaluator = dict()
        for key, module in modules.items():
            if isinstance(module, M.Module):
                self[key] = module
            else:
                self.non_metric_evaluator[key] = module

    @property
    def validate(self):
        return self._validate

    @validate.setter
    def validate(self, validate):
        self._validate = validate
        for key, evaluator in itertools.chain(self.items(), self.non_metric_evaluator.items()):
            evaluator.validate = validate

    def requires(self, forward: bool, channel: TMContractChannel=None, *args, **kwargs):

        schema_data = dict()
        for key, x in itertools.chain(self.items(), self.non_metric_evaluator.items()):
            schema_data.update(x.requires(forward=forward, channel=channel).data())
        return TMSchema(schema_data)

    def provides(self, forward: bool, channel: TMContractChannel=None, *args, **kwargs):

        schema_data = dict()
        for key, x in itertools.chain(self.items(), self.non_metric_evaluator.items()):
            schema_data.update(x.provides(forward=forward, channel=channel).data())
        return TMSchema(schema_data)

        
    def clone(self):
        """

        Returns:

        """
        cloned_modules = dict((key, x.clone())
                              for key, x in itertools.chain(self.items(), self.non_metric_evaluator.items()))

        return TMEvaluatorCollection(cloned_modules)

    def update(self, *args, **kwargs): 
        """

        Args:
            *args:
            **kwargs:

        Returns:

        """
        for key, evaluator in itertools.chain(self.items(), self.non_metric_evaluator.items()):
            evaluator.update(*args, **kwargs)


    def compute(self) -> Dict[str, Any]:
        """

        Returns:

        """
        result = dict()
        for key, evaluator in itertools.chain(self.items(), self.non_metric_evaluator.items()):
            eval_result = evaluator.compute()
            for k2, value in eval_result.items():
                result[f"{key}.{k2}"] = value

        return result

    def reset(self):
        """

        Returns:

        """
        for key, evaluator in itertools.chain(self.items(), self.non_metric_evaluator.items()):
            evaluator.reset()

    def to(self, device):
        for key, evaluator in itertools.chain(self.items(), self.non_metric_evaluator.items()):
            self[key].to(device)


class TMMultiEvaluator(TMEvaluatorCollection):
    """
    TMMultiEvaluator:
    multi-task data, each with a evaluator
    """

    def requires(self, forward: bool, channel: TMContractChannel=None, task=None, *args, **kwargs):

        if task is None:
            schema_dict = dict()
            for task, evaluator in itertools.chain(self.items(), self.non_metric_evaluator.items()):
                schema_dict[task] = evaluator.requires(forward, channel, *args, **kwargs).data()

            return TMSchema(schema_dict)
        else:
            evaluator = self[task] if task in self else self.non_metric_evaluator[task]
            return evaluator.requires(forward, channel, *args, **kwargs)

    def provides(self, forward: bool, channel: TMContractChannel=None, task=None, *args, **kwargs):

        if task is None:
            schema_dict = dict()
            for task, evaluator in itertools.chain(self.items(), self.non_metric_evaluator.items()):
                schema_dict[task] = evaluator.provides(forward, channel, *args, **kwargs).data()

            return TMSchema(schema_dict)
        else:
            evaluator = self[task] if task in self else self.non_metric_evaluator[task]
            return evaluator.provides(forward, channel, *args, **kwargs)

    def update(self, *args, **kwargs):  # pylint: disable=E0202
        """

        Args:
            *args:
            **kwargs:

        Returns:

        """

        machine, truth = args
        for key, evaluator in self.items():
            evaluator.update(machine[key], truth[key])

    def clone(self):
        """

        Returns:

        """
        cloned_modules = dict((key, x.clone()) for key, x in self.items())

        return TMMultiEvaluator(cloned_modules)

class TMContractedEvaluator(TMEvaluatorInterface, M.Module):
    """
    TMContractAdaptiveLoss
    """
    def __init__(self, evaluator: TMEvaluatorInterface,
                 truth_contract: Union[TMContract, dict]=None,
                 inference_contract: Union[TMContract, dict]=None):
        """

        Args:
            loss ():
            machine_adaptor ():
            truth_adaptor ():
        """
        TMEvaluatorInterface.__init__(self)
        M.Module.__init__(self)

        self.evaluator = evaluator

        self.truth_contract = TMContract(truth_contract)\
            if isinstance(truth_contract, dict) else truth_contract
        self.inference_contract = TMContract(inference_contract)\
            if isinstance(inference_contract, dict) else inference_contract

    def requires(self, forward: bool, channel: TMContractChannel = None, *args, **kwargs):

        if not forward:
            if channel is None:
                return TMChannelSchema({TMContractChannel.Truth: {},
                                           TMContractChannel.Inference: {}})
            else:
                return TMSchema({})
        #        assert forward, "loss does not require anything in backward procedure "

        if channel == TMContractChannel.Truth or channel is None:
            truth_schema = self.loss.ForwardRequestSchema[TMContractChannel.Truth]
            if self.truth_contract:
                truth_schema = TMSchema(self.truth_contract.backward(truth_schema.data()))

        if channel == TMContractChannel.Inference or channel is None:
            inference_schema = self.loss.ForwardRequestSchema[TMContractChannel.Inference]
            if self.inference_contract:
                inference_schema = TMSchema(self.inference_schema.backward(inference_schema.data()))

        if channel is None:
            return TMChannelSchema({TMContractChannel.Truth: truth_schema,
                                       TMContractChannel.Inference: inference_schema})
        elif channel == TMContractChannel.Truth:
            return truth_schema
        elif channel == TMContractChannel.Inference:
            return inference_schema
        else:
            raise Exception(f"unsupported channel: {channel}")

    def provides(self, forward: bool, channel: TMContractChannel = None, *args, **kwargs):

        if channel is None:
            return TMChannelSchema({TMContractChannel.Truth: {},
                                       TMContractChannel.Inference: {}})
        else:
            return TMSchema({})

    def update(self, predict, truth):  # pylint: disable=E0202
        """
        Iteratively call update for each metric. Positional arguments (args) will
        be passed to every metric in the collection, while keyword arguments (kwargs)
        will be filtered based on the signature of the individual metric.
        """
        predict = self.inference_contract.forward(predict) if self.inference_contract else predict
        truth = self.truth_contract.forward(truth) if self.truth_contract else truth

        self.evaluator.update(predict, truth)

    def compute(self):

        return self.evaluator.compute()

    def metric_names(self):

        return self.evaluator.metric_names()

    def clone(self):

        return TMContractedEvaluator(self.evaluator.clone(), self.truth_contract, self.inference_contract)

    def reset(self):

        self.evaluator.reset()

    def to(self, device):
        self.evaluator.to(device)


class TMSupportEvaluatorMixin(object):

    Evaluator = None

    EvaluatorTruthContract = None
    EvaluatorInferenceContract = None


    @classmethod
    def create_evaluator(cls, evaluator, hyper_params, truth_contract, inference_contract):
        if isinstance(evaluator, TMEvaluatorInterface):
            pass
        elif inspect.isclass(evaluator) and issubclass(evaluator, TMEvaluatorInterface) and \
                issubclass(evaluator, TMSerializableComponent):
            evaluator.init_class()
            evaluator = evaluator.create(hyper_params)
        else:
            raise Exception(f"unknown evaluator setting {evaluator}")

        truth_contract = TMContract.parse(truth_contract)
        inference_contract = TMContract.parse(inference_contract)

        if truth_contract or inference_contract:
            evaluator = TMContractedEvaluator(evaluator,
                                                      truth_contract=truth_contract,
                                                      inference_contract=inference_contract)
        return evaluator

    def __init__(self, hyper_params, default_init=True):

        self.evaluator = None


        if self.Evaluator is None:
            return

        if not default_init:
            return

        if not isinstance(self.Evaluator, dict):

            self.evaluator = self.create_evaluator(self.Evaluator, hyper_params,
                                                   self.EvaluatorTruthContract,
                                                   self.EvaluatorInferenceContract)
        else:
            evaluators = dict()
            for key, evaluator in self.Evaluator.items():
                evaluators[key] = self.create_evaluator(evaluator, hyper_params[key],
                                                   self.EvaluatorTruthContract[key],
                                                   self.EvaluatorInferenceContract[key])
            self.evaluator = TMEvaluatorCollection(evaluators)

    # @property
    # def evaluator(self):
    #     return self.evaluator


    @classmethod
    def provide_evaluator(cls, contract_graph: TMContractGraph, controller, controller_role):


        if cls.Evaluator is not None:

            evaluator_role = f"{controller_role}Evaluator"

            evaluator_classes = dict()
            contracts = dict()
            if isinstance(cls.Evaluator, dict):
                for key, evaluator in cls.Evaluator.items():
                    role = f"{evaluator_role}.{key}"
                    evaluator_classes[role] = evaluator if inspect.isclass(evaluator) \
                        else evaluator.__class__
                    contracts[role] = (TMContract.parse(cls.EvaluatorTruthContract[key]),
                                      TMContract.parse(cls.EvaluatorInferenceContract[key]))
            else:
                evaluator_classes[evaluator_role] = cls.Evaluator if inspect.isclass(cls.Evaluator) \
                        else cls.Evaluator.__class__
                contracts[evaluator_role] = (TMContract.parse(cls.EvaluatorTruthContract),
                                            TMContract.parse(cls.EvaluatorInferenceContract))

            for role, evaluator_class in evaluator_classes.items():

                evaluator_class.attach(contract_graph, role=evaluator_role)

                truth_contract, inference_contract = contracts[role]

                channel_mapping = {TMContractChannel.Target: TMContractChannel.Truth,
                                           TMContractChannel.Inference: TMContractChannel.Inference}
                contract = {TMContractChannel.Truth: truth_contract,
                                    TMContractChannel.Inference: inference_contract}

                contract_graph.connect_consumer(component=controller, component_role=controller_role,
                                                  consumer=evaluator_class,
                                                  consumer_role=evaluator_role,
                                                  channel_mapping=channel_mapping,
                                                  contract=contract,
                                                  )

                yield f"{evaluator_class.__name__}@{evaluator_role}"



    def states(self):
        states = super().states()
        if self.evaluator:
            states["evaluator"] = self.evaluator.states()
        return states

    def load_states(self, states):
        if self.evaluator is not None and "evaluator" in states:
            self.evaluator.load_states(states["evaluator"])

        super().load_states(states)


"""
Design:

The major problem of two level evaluation (problem level and raw level) is that, we need to reuse the stream without
caching all the inferenced results in the memory.

One option is to use batch wise operation, not global stream with operation. See commit 1e79eff for this implementation.
However, it is intrusive to obtain many details in the operator and modeler.

Current implementation solve the problem by re-create the problem stream after the problem evaluation,
and the raw evaluator can use that stream for evaluation.
A drawback of this approach is that the problem evaluation streategy does not know when the stream is iterated
and thus when the performance is ready. So an extra singal ON_EVALUATION_TERMINATING is added to notify the problem
evaluation strategy that the stream has been iterated and the performance is ready.
"""
import sys
from collections import namedtuple, defaultdict

from tripmaster.core.concepts.evaluator import TMEvaluationStrategy
from tripmaster.utils.stream import isolate_iterators

from dataclasses import dataclass

@dataclass
class EvaluationStreamInfo:
    """
    EvaluationMachineStreamInfo
    """

    truth_stream: object = None
    inferenced_stream: object = None
    local_rank: int = None
    device: object = None
    epoch: int = None
    step: int  = None

@dataclass
class MachineEvaluationStreamInfo(EvaluationStreamInfo):
    objective_stream: object = None

@dataclass
class EvaluationResults:
    performance: float = None
    local_rank: int = None
    epoch: int = None
    step: int = None

@dataclass
class MachineEvaluationResults(EvaluationResults):

    objective: dict = None

@dataclass
class EvaluationTerminatingInfo:
    local_rank: int = None
    epoch: int = None
    step: int = None

@dataclass
class InferenceStreamInferencedInfo:
    stream: object = None
    local_rank: int = None


class TMMachineEvaluationStrategy(TMEvaluationStrategy):
    """
    DefaultTMProblemEvaluationStrategy
    """

    def __init__(self, evaluator: TMEvaluator):
        super().__init__(evaluator=evaluator)

        self.channel_objective = dict()

    def evaluate_machine_channel(self, loss_channel, truth_machine_channel, inference_machine_channel,
                                 channel, device):

        evaluator = self.get_evaluator(channel, device)
        total_loss = 0
        sample_num = 0

        for objective, machine_truth, machine_inference in zip(loss_channel, truth_machine_channel,
                                                          inference_machine_channel):
            if evaluator.validate:

                assert evaluator.requires(forward=True,
                                        channel=TMContractChannel.Truth).is_valid(machine_truth)
                assert evaluator.requires(forward=True,
                                        channel=TMContractChannel.Inference).is_valid(machine_inference)

            evaluator.update(machine_inference, machine_truth)

            total_loss += objective["objective"] * objective["sample_num"]
            sample_num += objective["sample_num"]

            yield objective, machine_truth, machine_inference

        self.channel_performance[channel] = evaluator.compute()
        self.channel_objective[channel] = total_loss / sample_num if sample_num > 0 else 0.0

    def on_stream_inferenced(self, info: MachineEvaluationStreamInfo):
        """
        Args:
            inference_stream:
            scenario:
            local_rank:
        Returns:
        """

        for channel in info.truth_stream.eval_channels:
            objective_channel, truth_problem_channel, inference_problem_channel = isolate_iterators(
                self.evaluate_machine_channel(info.objective_stream[channel],
                                              info.truth_stream[channel],
                                              info.inferenced_stream[channel],
                                              channel, info.device),
                3
            )
            info.objective_stream[channel] = objective_channel
            info.truth_stream[channel] = truth_problem_channel
            info.inferenced_stream[channel] = inference_problem_channel

        return info

    def on_evaluation_end(self, info: EvaluationTerminatingInfo):

        machine_performance = MachineEvaluationResults(
            objective=self.channel_objective, performance=self.channel_performance,
            local_rank=info.local_rank, epoch=info.epoch)

        return machine_performance


class TMProblemEvaluationStrategy(TMEvaluationStrategy):
    """
    DefaultTMProblemEvaluationStrategy
    """

    def __init__(self, evaluator: TMEvaluator):
        super().__init__(evaluator=evaluator)

    def evaluate_problem_channel(self, truth_problem_channel, inference_problem_channel, channel, device):

        evaluator = self.get_evaluator(channel, device)
        for problem_truth, problem_inference in zip(truth_problem_channel,
                                                    inference_problem_channel):
            if evaluator.validate:

                assert evaluator.requires(forward=True,
                                        channel=TMContractChannel.Truth).is_valid(problem_truth)
                assert evaluator.requires(forward=True,
                                        channel=TMContractChannel.Inference).is_valid(problem_inference)

            evaluator.update(problem_inference, problem_truth)

            yield problem_truth, problem_inference

        self.channel_performance[channel] = evaluator.compute()

    def on_stream_inferenced(self, info: EvaluationStreamInfo):
        """
        Args:
            inference_stream:
            scenario:
            local_rank:
        Returns:
        """
        logger.info("inferenced info receivied")

        for channel in info.truth_stream.eval_channels:
            truth_problem_channel, inference_problem_channel = isolate_iterators(
                self.evaluate_problem_channel(info.truth_stream[channel],
                                              info.inferenced_stream[channel],
                                              channel, info.device),
                2
            )
            info.truth_stream[channel] = truth_problem_channel
            info.inferenced_stream[channel] = inference_problem_channel

        return info

    def on_evaluation_end(self, info: EvaluationTerminatingInfo):

        problem_performance = EvaluationResults(
            performance=self.channel_performance, local_rank=info.local_rank, epoch=info.epoch)

        return problem_performance


class TMTaskEvaluationStrategy(TMEvaluationStrategy):
    """
    TMDefaultTaskEvaluationStrategy
    """

    def __init__(self, evaluator: TMEvaluator):

        super().__init__(evaluator=evaluator)

    def evaluate_task_channel(self, truth_task_channel, inference_task_channel, channel, device):

        evaluator = self.get_evaluator(channel, device)
        for task_truth, task_inference in zip(truth_task_channel,
                                              inference_task_channel):

            if evaluator.validate:

                assert evaluator.requires(forward=True,
                                        channel=TMContractChannel.Truth).is_valid(task_truth)
                assert evaluator.requires(forward=True,
                                        channel=TMContractChannel.Inference).is_valid(task_inference)

            evaluator.update(task_inference, task_truth)

            yield task_truth, task_inference

        self.channel_performance[channel] = evaluator.compute()

    def on_stream_inferenced(self, info: EvaluationStreamInfo):

        logger.info("inferenced info receivied")

        for channel in info.truth_stream.eval_channels:
            truth_problem_channel, inference_problem_channel = isolate_iterators(
                self.evaluate_task_channel(info.truth_stream[channel],
                                           info.inferenced_stream[channel],
                                           channel, info.device),
                2
            )
            info.truth_stream[channel] = truth_problem_channel
            info.inferenced_stream[channel] = inference_problem_channel

        return info

    def on_evaluation_end(self, info: EvaluationTerminatingInfo):

        task_performance = EvaluationResults(
            performance=self.channel_performance, local_rank=info.local_rank, epoch=info.epoch)

        return task_performance