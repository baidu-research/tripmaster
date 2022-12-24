import weakref
from typing import Union
from tqdm import tqdm
import abc
from tripmaster.core.components.evaluator import MachineEvaluationStreamInfo
from tripmaster.core.components.machine.machine import TMMultiMachine
from tripmaster.core.components.modeler.machine_memory import TMMachine2MemoryModeler, \
    TMProtoMultiMachine2MemoryModeler
from tripmaster.core.components.modeler.memory_batch import TMMemory2BatchModeler
from tripmaster.core.components.operator.strategies.event_trigger import TMEpochwiseTrigger
from tripmaster.core.components.operator.strategies.metric_logging import TMMetricLoggingStrategyFactory
from tripmaster.core.components.operator.strategies.model_selection import BestOneModelSelectionStrategy
from tripmaster.core.concepts.component import TMConfigurable, TMSerializableComponent
from tripmaster.core.concepts.contract import TMContractChannel
from tripmaster.core.concepts.data import TMDataStream, TMDataLevel, TMMultiDataStream

from tripmaster.core.concepts.operator import TMOperatorInterface
from tripmaster.core.components.operator.strategies.distributed import TMDistributedStrategyFactory

from tripmaster.core.components.operator.strategies.metric_logging import TMMetricLoggingStrategyFactory
from tripmaster.core.concepts.contract import TMContractChannel
from tripmaster.core.components.operator.strategies.optimization import EpochwiseLRUpdateStrategy

from tripmaster.core.concepts.data import TMDataStream, TMDataLevel

from tripmaster import logging
from tripmaster.core.concepts.scenario import TMScenario
from tripmaster.utils.stream import isolate_iterators
from tqdm import tqdm
from tripmaster import P, T, M, D


logger = logging.getLogger(__name__)


def deep_merge_dict(dict1, dict2):
    """
    merge fields in dict2 into dict1
    Args:
        dict1:
        dict2:

    Returns:

    """
    assert isinstance(dict1, dict) and isinstance(dict2, dict)
    for key in dict2.keys():
        if key not in dict1:
            dict1[key] = dict2[key]
            continue

        if not isinstance(dict1[key], dict) or not isinstance(dict2[key], dict):
            continue

        assert isinstance(dict1[key], dict) and isinstance(dict2[key], dict)

        deep_merge_dict(dict1[key], dict2[key])

from blinker import signal

class TMOperator(TMOperatorInterface):
    """
    TMOperatorBase
    """

    Machine2MemoryModeler = TMMachine2MemoryModeler
    Memory2BatchModeler = TMMemory2BatchModeler


    def __init__(self, hyper_params, machine=None, scenario=None, states=None, **kwargs):
        self.scenario = scenario
        super().__init__(hyper_params, machine=machine, scenario=scenario, states=states, **kwargs)


    def runtime(self, runtime_options):

        self.select_distributed_strategy(runtime_options)
        self.build_operator_modeler(runtime_options)
        self.select_metric_logging_strategy(runtime_options)
    def build_operator_modeler(self, runtime_options):

        if self.machine.DataTraits and self.machine.DataTraits.SAMPLE_OOM_POSSIBLE:
            resource_limit = runtime_options.resource.memory.learning_memory_limit
        else:
            resource_limit = None

        self.memory_modeler = self.Machine2MemoryModeler(self.machine.DataTraits, resource_limit)
        # if isinstance(self.machine, TMMultiMachine):
        #     class ThisModelerType(TMProtoMultiMachine2MemoryModeler):
        #         ProtoType = self.Machine2MemoryModeler
        #
        #     ThisModelerType.init_class()
        #
        #     self.memory_modeler = ThisModelerType(None)


        self.batch_modeler = self.Memory2BatchModeler(hyper_params=runtime_options,
                                                 sample_traits=self.machine.DataTraits,
                                                 batch_traits=self.machine.BatchTraits)


    @abc.abstractmethod
    def fit_memory(self, machine_samplestream: TMDataStream, scenario: TMScenario):
        pass

    @abc.abstractmethod
    def unfit_memory(self, memory_samplestream: TMDataStream, scenario: TMScenario,
                     with_truth=False):
        pass

    @abc.abstractmethod
    def batchify(self, memory_samplestream: TMDataStream, scenario: TMScenario):
        pass

    @abc.abstractmethod
    def unbatchify(self, memory_batchstream: TMDataStream, scenario: TMScenario,
                   with_truth=False):
        pass

    def select_distributed_strategy(self, runtime_options):
        """

        Returns:

        """
        resource = runtime_options.resource
        distributed = runtime_options.distributed

        if not TMDistributedStrategyFactory.get().has_strategy(distributed):
            raise Exception(f"distributed strategy {distributed} not supported")

        distributed_strategy_class = TMDistributedStrategyFactory.get().choose(distributed)
        
        if resource.computing.gpu_per_trial <= 0:
            world_size = resource.computing.cpu_per_trial
            use_gpu = False
        else:
            world_size = resource.computing.gpu_per_trial
            use_gpu = True

        self.distributed_strategy = distributed_strategy_class(operator=self,
                                   world_size=world_size, use_gpu=use_gpu)

    def select_metric_logging_strategy(self, runtime_options):

        metric_logging_strategy_name = runtime_options.metric_logging.type

        metric_logging_strategy_class = TMMetricLoggingStrategyFactory().get().choose(metric_logging_strategy_name)
        self.metric_logging_strategy = metric_logging_strategy_class(runtime_options.metric_logging.strategies[metric_logging_strategy_name])


    def device(self, local_rank):

        use_gpu = self.distributed_strategy.use_gpu
        if use_gpu:
            device = f"{D.device_prefix()}:{local_rank}"
        else:
            device = "cpu"

        return device

    def reallocate_data(self, data, local_rank):
        """
        reallocate data to current working device determined by local_rank
        """

        use_gpu = self.distributed_strategy.use_gpu
        if use_gpu:
            device = f"{D.device_prefix()}:{local_rank}"
            data = self.machine.BatchTraits.to_device(data, device)
        return data



class TMEvaluatorMixin(TMConfigurable):


    evaluate_signal = signal("evaluate")

    @abc.abstractmethod
    def evaluate(self, source, local_rank, epoch, step):
        pass

class TMInferenceMixIn(TMConfigurable):
    @abc.abstractmethod
    def inference(self, source, runtime_options, local_rank):
        pass

class TMLearnerMixin(TMConfigurable):
    """
    TMLearnerMixin: Add learning ability to TMOperator
    """
    Optimizer = None
    LRScheduler = None
    OptimizationStrategy = EpochwiseLRUpdateStrategy
    ModelSelectionStrategy = BestOneModelSelectionStrategy
    EvaluationTriggerStrategy = TMEpochwiseTrigger

    good_model_discovered_signal = signal("good_model_discovered")

    def __init__(self, hyper_params, **kwargs):

        super().__init__(hyper_params, **kwargs)

        self.algorithm_params = hyper_params.optimizer.algorithm
        self.lr_scheduler_params = hyper_params.optimizer.lr_scheduler
        self.gradient_clip_val = hyper_params.optimizer.gradient_clip_val

        self.check_optimizer_lrsheduler_setting()
        self.create_optimizer(self.machine)

        optimizer_strategy_conf = hyper_params.optimizer.strategy
        self.optimization_strategy = self.OptimizationStrategy(
            optimizer_strategy_conf,
            optimizer=self.optimizer, lr_scheduler=self.lr_scheduler,
            gradient_clip_val=self.gradient_clip_val
        )

        self.model_selector = self.ModelSelectionStrategy(hyper_params=self.hyper_params.modelselector,
                                                          )

        self.metric_logger = None

        self.evaluation_trigger = self.EvaluationTriggerStrategy(
            hyper_params=self.hyper_params.evaluator_trigger)

        self.epoch = -1
        self.step = -1

    def states(self):

        return {"epoch": self.epoch, "step": self.step,
                "model_selector": self.model_selector.states(),
                "evaluation_trigger": self.evaluation_trigger.states(),
                "optimization_strategy": self.optimization_strategy.states()}

    def load_states(self, states):

        self.epoch = states["epoch"]
        self.step = states["step"]
        self.model_selector.load_states(states["model_selector"])
        self.optimization_strategy.load_states(states["optimization_strategy"])

    def check_optimizer_lrsheduler_setting(self):

        if isinstance(self.Optimizer, dict) and \
                list(self.Optimizer.keys()) != list(self.algorithm_params.keys()):
            message = f"Multiple optimizers {list(self.Optimizer.keys())} are required, \
                                but {list(self.algorithm_params.keys())} parameters are given"
            raise Exception(message)

        if isinstance(self.LRScheduler, dict) and \
                list(self.LRScheduler.keys()) != list(self.lr_scheduler_params.keys()):
            message = f"Multiple lr_schedulers {list(self.LRScheduler.keys())} are required, \
                                but {list(self.lr_scheduler_params.keys())} parameters are given"
            raise Exception(message)

        if isinstance(self.LRScheduler, dict) and isinstance(self.LRScheduler, dict) and \
                list(self.LRScheduler.keys()) != list(self.LRScheduler.keys()):
            message = f"Optimizers and lr_schedulers should have same submodules, but \
                                {list(self.LRScheduler.keys())} for optimizer and \
                                but {list(self.LRScheduler.keys())} for lr_sheduler are given"
            raise Exception(message)

    def check_optimizer_submodule_match(self, submodules):
        valid_submoudle_keys = set()
        for key, modules in submodules.items():
            parameters = sum([list(m.parameters()) for m in modules], [])
            if len(parameters) > 0:
                valid_submoudle_keys.add(key)

        if not set(self.Optimizer.keys()) >= set(valid_submoudle_keys):
            message = f"The machine requires optimizer for {list(valid_submoudle_keys)}, \
                                        but {list(self.Optimizer.keys())} are provided"
            raise Exception(message)

        useless_config_keys = set(self.Optimizer.keys()) - set(valid_submoudle_keys)
        for key in useless_config_keys:
            if self.algorithm_params[key] is not None:
                logger.warning(f"Useless optimization algorithm configure {key} found")

        if not set(self.LRScheduler.keys()) >= set(valid_submoudle_keys):
            message = f"The machine requires lr_schedulers for {list(valid_submoudle_keys)}, \
                                        but {list(self.LRScheduler.keys())} are provided"
            raise Exception(message)

        useless_config_keys = set(self.LRScheduler.keys()) - set(valid_submoudle_keys)
        for key in useless_config_keys:
            if self.algorithm_params[key] is not None:
                logger.warning(f"Useless optimization algorithm configure {key} found")

    def create_optimizer(self, machine):

        submodules = machine.submodules()

        if not isinstance(self.Optimizer, dict):
            self.Optimizer = dict((key, self.Optimizer) for key in submodules.keys())
            self.algorithm_params = dict((key, self.algorithm_params) for key in submodules.keys())

        if not isinstance(self.LRScheduler, dict):
            self.LRScheduler = dict((key, self.LRScheduler) for key in submodules.keys())
            self.lr_scheduler_params = dict((key, self.lr_scheduler_params) for key in submodules.keys())

        self.check_optimizer_submodule_match(submodules)

        self.optimizer = dict()
        self.lr_scheduler = dict()

        for name, modules in submodules.items():

            parameters = sum([list(m.parameters()) for m in modules], [])
            if len(parameters) == 0:
                continue

            if self.Optimizer[name] is None:
                for module in modules:
                    module.requires_grad_(False)

                continue

            self.optimizer[name], self.lr_scheduler[name] = P.OptimizerBehaviors.create_optimization_components(
                parameters,
                self.Optimizer[name], self.algorithm_params[name],
                self.LRScheduler[name], self.lr_scheduler_params[name],
                self.gradient_clip_val
            )

    def test(self, test_config):

        self.optimization_strategy.test(test_config)

    def eval_and_select_model(self, train_batchstreams: TMDataStream, local_rank, epoch, step):

        # if local_rank != 0:
        #     return

        signal_returns = self.evaluate(train_batchstreams, local_rank, epoch, step)
        assert len(signal_returns) == 1
        evaluation_results = signal_returns[0][1]

        self.metric_logging_strategy.log(evaluation_results)

        self.optimization_strategy.on_evaluated(self.machine, evaluation_results)

        model_info = self.model_selector.select_model(evaluation_results, self.machine, self)
        if model_info:
            self.good_model_discovered_signal.send(model_info)

    @abc.abstractmethod
    def train(self, source, runtime_options, local_rank):
        pass

