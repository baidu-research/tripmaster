"""
type definitions for cfl
"""
import abc
import weakref
from collections import defaultdict
from inspect import isclass
from typing import Optional, List, Union

from tripmaster import logging
from tripmaster.core.components.machine.machine import TMSharedMultiTaskMachine, TMMultiMachine
from tripmaster.core.components.operator.strategies.model_selection import SelectedModelInfo
from tripmaster.core.components.problem import TMMultiProblem
from tripmaster.core.components.task import TMMultiTask

from tripmaster.core.concepts.component import TMSerializableComponent, TMConfigurable
from tripmaster.core.concepts.contract import TMContract, TMContract, TMMultiContract, TMContractInterface
from tripmaster.core.components.operator.operator import TMLearnerMixin
from tripmaster.core.concepts.data import TMDataStream, TMDataLevel

from tripmaster.core.components.modeler.modeler import TMContractOnlyModeler, TMSharedModeler, TMMultiModeler
from tripmaster.core.concepts.hyper_params import TMHyperParams
from tripmaster.core.concepts.scenario import TMScenario

from tripmaster.core.system.validation import TMSystemValidator
from tripmaster.utils.function import return_none

from tripmaster.utils.profile_support import profile

from collections import OrderedDict
from tripmaster.core.app.config import TMConfig
import addict
import os
logger = logging.getLogger(__name__)


#
# class TMSystemSignal(object):
#
#     ON_TASK_DATASTREAM_CREATED = "on_task_datastream_created"
#     ON_PROBLEM_DATASTREAM_CREATED = "on_problem_datastream_created"

def to_save(config):
    """
    check whether the config requests to save
    """
    return (config and config.serialize and config.serialize.save)


def to_load(config):
    """
    check whether the config requests to save
    """
    return (config and config.serialize and config.serialize.load)


class TMSystemRuntimeCallbackInterface(object):
    """
    TMSystemCallback
    """



    def on_task_data_loaded(self, task_data):
        pass

    def on_problem_data_loaded(self, problem_data):
        pass

    def on_problem_data_built(self, problem_data):
        pass

    def on_machine_data_loaded(self, machine_data):
        pass

    def on_machine_data_built(self, machine_data):
        pass

    def on_memory_data_loaded(self, memory_data):
        pass

    def on_memory_data_built(self, memory_data):
        pass

    def on_data_phase_finished(self, system):
        pass

    def on_operation_finished(self, system):
        pass




class TMSystem(TMSerializableComponent):
    """
    TMSystem
    """

    TaskType = None
    TPModelerType = None
    ProblemType = None
    PMModelerType = None
    MachineType = None
    OperatorType = None

    Task2ModelerContract = None
    Modeler2ProblemContract = None
    Task2ProblemContract = None
    Problem2ModelerContract = None
    Modeler2MachineContract = None
    Problem2MachineContract = None

    def pre_system_creation(self):
        pass

    def post_system_creation(self):
        pass

    def pre_task_creation(self):
        pass

    def post_task_creation(self):
        pass

    def on_task_ready(self):
        pass

    def pre_tpmodeler_creation(self):
        pass

    def post_tpmodeler_creation(self):
        pass

    def on_tpmodeler_ready(self):
        pass

    def pre_problem_creation(self):
        pass

    def post_problem_creation(self):
        pass

    def on_problem_ready(self):
        pass

    def pre_pmmodeler_creation(self):
        pass

    def post_pmmodeler_creation(self):
        pass

    def on_pmmodeler_ready(self):
        pass

    def pre_machine_creation(self):
        pass

    def post_machine_creation(self):
        pass

    def on_machine_ready(self):
        pass

    def pre_operator_creation(self):
        pass

    def post_operator_creation(self):
        pass

    def on_operator_ready(self):
        pass



    @classmethod
    def init_class(cls):

        if cls.TaskType is not None:
            cls.TaskType.init_class()

        if cls.TPModelerType is not None:
            cls.TPModelerType.init_class()

        if cls.ProblemType is not None:
            cls.ProblemType.init_class()

        if cls.PMModelerType is not None:
            cls.PMModelerType.init_class()

        if cls.MachineType is not None:
            cls.MachineType.init_class()

        if cls.OperatorType is not None:
            cls.OperatorType.init_class()


    def __init__(self, hyper_params,
                 callbacks: Optional[List[TMSystemRuntimeCallbackInterface]] = None,
                 states=None,
                 task=None, tp_modeler=None, problem=None, pm_modeler=None, machine=None, operator=None
                 ):
        super().__init__(hyper_params=hyper_params)

        self.callbacks = callbacks if callbacks else []
        self.operator_callbacks = []

        self.test_config = None
        self.provided_states = states

        if self.provided_states is None:
            self.provided_states = defaultdict(return_none)

        self.task = task
        self.tp_modeler = tp_modeler
        self.problem = problem
        self.pm_modeler = pm_modeler
        self.machine = machine
        self.operator = operator
        #
        # if self.task:
        #     self.hyper_params.task = self.task.hyper_params
        # if self.tp_modeler:
        #     self.hyper_params.tp_modeler = self.tp_modeler.hyper_params
        # if self.problem:
        #     self.hyper_params.problem = self.problem.hyper_params
        # if self.pm_modeler:
        #     self.hyper_params.pm_modeler = self.pm_modeler.hyper_params
        # if self.machine:
        #     self.hyper_params.machine = self.machine.hyper_params
        # if self.operator:
        #     if self.is_learning():
        #         self.hyper_params.learner = self.operator.hyper_params
        #     else:
        #         self.hyper_params.inferencer = self.operator.hyper_params

    def operator_from_scratch(self):

        self.provided_states["machine"] = None
        self.provided_states["operator"] = None


    def build_data_pipeline(self):

        if self.TaskType:
            self.pre_task_creation()
            self.task = self.TaskType.create(self.hyper_params.task, states=self.provided_states["task"])
            self.post_task_creation()

        self.pre_tpmodeler_creation()
        if self.TPModelerType is None:
            if self.Task2ProblemContract is None:
                self.tp_modeler = None
            else:
                self.tp_modeler = TMContractOnlyModeler(hyper_params=None)
                self.task_problem_contract = self.parse_contract(self.Task2ProblemContract,
                                                                     self.hyper_params.contracts.task2problem)
                self.tp_modeler.set_contract(self.task_problem_contract, None)

        else:
            self.tp_modeler = self.TPModelerType.create(self.hyper_params.tp_modeler,
                                                        states=self.provided_states["tp_modeler"])

            self.task_modeler_contract = self.parse_contract(self.Task2ModelerContract,
                                                             self.hyper_params.contracts.task2modeler)
            self.modeler_problem_contract = self.parse_contract(self.Modeler2ProblemContract,
                                                                self.hyper_params.contracts.modeler2problem)
            self.tp_modeler.set_contract(self.task_modeler_contract, self.modeler_problem_contract)

        self.post_tpmodeler_creation()

        if self.ProblemType:
            self.pre_problem_creation()
            self.problem = self.ProblemType.create(self.hyper_params.problem, states=self.provided_states["problem"])
            self.post_problem_creation()

        self.pre_pmmodeler_creation()
        if self.PMModelerType is None:

            if self.Problem2MachineContract is None:
                self.pm_modeler = None
            else:
                self.pm_modeler = TMContractOnlyModeler(hyper_params=None)
                self.problem_machine_contract = self.parse_contract(self.Problem2MachineContract,
                                                                     self.hyper_params.contracts.problem2machine)
                self.pm_modeler.set_contract(self.problem_machine_contract, None)

        else:

            self.pm_modeler = self.PMModelerType.create(self.hyper_params.pm_modeler,
                                                        states=self.provided_states["pm_modeler"])
            self.problem_modeler_contract = self.parse_contract(self.Problem2ModelerContract,
                                                                 self.hyper_params.contracts.problem2modeler)
            self.modeler_machine_contract = self.parse_contract(self.Modeler2MachineContract,
                                                                 self.hyper_params.contracts.modeler2machine)
            self.pm_modeler.set_contract(self.problem_modeler_contract,
                                          self.modeler_machine_contract)
        self.post_pmmodeler_creation()

    def lazy_build_machine_operator(self):

        self.pre_machine_creation()

        machine_states = self.provided_states.get("machine", None)

        self.machine = self.MachineType.create(self.hyper_params.machine,
                                               states=machine_states)

        self.post_machine_creation()

        self.pre_operator_creation()
        operator_hparam = self.hyper_params.learner if self.is_learning() else self.hyper_params.inferencer

        if self.is_learning() and "learner" in self.provided_states:
            operator_states = self.provided_states["learner"]
        elif not self.is_learning() and "inferencer" in self.provided_states:
            operator_states = self.provided_states["inferencer"]
        else:
            operator_states = None

        self.operator = self.OperatorType.create(operator_hparam,
                                                 states=operator_states,
                                                 machine=self.machine)

        self.post_operator_creation()

        self.post_system_creation()

        self.build_evaluation_pipeline()


    def states(self):

        states = super().states()

        if "operator" in states:
            if self.is_learning():
                states["learner"] = states["operator"]
            else:
                states["inferencer"] = states["operator"]

            del states["operator"]
        
        for key in self.provided_states:
            if key not in states or states[key] is None:
                states[key] = self.provided_states[key]

        return states

    def load_states(self, states):
        if self.is_learning() and "learner" in states:
            states["operator"] = states["learner"]
        elif not self.is_learning() and "inferencer" in states:
            states["operator"] = states["inferencer"]

        super().load_states(states)

    # def update_machine_operator_hyperparams(self, datastream):
    #     """
    #     update machines hyperparams according to system and data
    #     Returns:
    #
    #     """
    #     pass

    # def update_machine_tpmodeler_hyperparams(self, machine_params, tp_modeler):
    #     """
    #     update machines hyperparams according to tpmodeler
    #     Returns:
    #
    #     """
    #     pass

    def test(self, config):

        self.test_config = config

    def lazy_update_test_setting(self):

        if self.test_config is None:
            return

        if self.test_config.validate:
            self.validate = True

        if self.task is not None:
            self.task.test(self.test_config)
            self.task.validate = self.validate

        if self.tp_modeler is not None:
            self.tp_modeler.test(self.test_config)
            self.tp_modeler.validate = self.validate

        if self.problem is not None:
            self.problem.test(self.test_config)
            self.problem.validate = self.validate

        if self.pm_modeler is not None:
            self.pm_modeler.test(self.test_config)
            self.pm_modeler.validate = self.validate

        if self.machine is not None:
            self.machine.test(self.test_config)
            self.machine.validate = self.validate
            self.operator.test(self.test_config)
            self.operator.validate = self.validate

    @classmethod
    def is_learning(self):
        return issubclass(self.OperatorType, TMLearnerMixin)

    def parse_contract(self, x, config):
        """

        Args:
            x:

        Returns:

        """
        if x is None:
            return None
        elif isinstance(x, dict):
            return TMContract.parse(x)
        elif isinstance(x, TMContractInterface):
            return x
        elif isclass(x) and issubclass(x, TMContractInterface):
            assert isinstance(config, dict)
            return x(config)
        else:
            raise Exception("unknown contract")

    def build_evaluation_pipeline(self):
        from tripmaster.core.components.evaluator import TMMachineEvaluationStrategy, \
            TMProblemEvaluationStrategy, TMTaskEvaluationStrategy

        if self.machine.evaluator:
            self.machine_evaluating_strategy = TMMachineEvaluationStrategy(evaluator=self.machine.evaluator)
        else:
            self.machine_evaluating_strategy = None

        if self.problem and self.problem.evaluator:
            self.problem_evaluating_strategy = TMProblemEvaluationStrategy(evaluator=self.problem.evaluator)
        else:
            self.problem_evaluating_strategy = None

        if self.task and self.task.evaluator:
            self.task_evaluating_strategy = TMTaskEvaluationStrategy(evaluator=self.task.evaluator)
        else:
            self.task_evaluating_strategy = None

    @profile
    def evaluate_callback(self, machine_inference_info):

        from tripmaster.core.components.evaluator import MachineEvaluationStreamInfo, \
            EvaluationTerminatingInfo, \
            EvaluationStreamInfo

        assert isinstance(machine_inference_info, MachineEvaluationStreamInfo)

        eval_results = {}

        evalation_termination_info = EvaluationTerminatingInfo(
            local_rank=machine_inference_info.local_rank,
            epoch=machine_inference_info.epoch,
            step=machine_inference_info.step
        )

        last_inference_info = None
        last_strategy = None
        if self.machine_evaluating_strategy:
            self.machine_evaluating_strategy.on_evaluation_begin()
            machine_inference_info = self.machine_evaluating_strategy.on_stream_inferenced(machine_inference_info)
#            machine_evaluation_signal = signal(ON_MACHINE_STREAM_INFERENCED)
#            machine_inference_info = sequential_activate_for_stream(machine_evaluation_signal, machine_inference_info)
            last_inference_info = machine_inference_info
            last_strategy = self.machine_evaluating_strategy

        if self.problem_evaluating_strategy or self.task_evaluating_strategy:

            truth_stream = self.operator.unbatchify(
                machine_inference_info.truth_stream, scenario=TMScenario.Evaluation,
                with_truth=True)
            truth_stream = self.operator.unfit_memory(truth_stream, scenario=TMScenario.Evaluation,
                                                      with_truth=True)
            inference_stream = self.operator.unbatchify(
                machine_inference_info.inferenced_stream, scenario=TMScenario.Evaluation,
                with_truth=False)
            inference_stream = self.operator.unfit_memory(inference_stream,
                                                          scenario=TMScenario.Evaluation,
                                                          with_truth=False)

            if self.pm_modeler is not None:
                truth_problem_stream = self.pm_modeler.reconstruct_datastream(
                    truth_stream, scenario=TMScenario.Evaluation, with_truth=True)
                inference_problem_stream = self.pm_modeler.reconstruct_datastream(
                    inference_stream, scenario=TMScenario.Evaluation, with_truth=False)
            else:
                truth_problem_stream = truth_stream
                truth_problem_stream.level = TMDataLevel.reconstruct(truth_stream.level)
                inference_problem_stream = inference_stream
                inference_problem_stream.level = TMDataLevel.reconstruct(inference_stream.level)

            problem_inference_info = EvaluationStreamInfo(
                truth_stream=truth_problem_stream,
                inferenced_stream=inference_problem_stream,
                local_rank=machine_inference_info.local_rank,
                epoch=machine_inference_info.epoch,
                step=machine_inference_info.step,
                device=machine_inference_info.device
            )

            if self.problem_evaluating_strategy:
                self.problem_evaluating_strategy.on_evaluation_begin()
                problem_inference_info = self.problem_evaluating_strategy.on_stream_inferenced(problem_inference_info)

                last_inference_info = problem_inference_info
                last_strategy = self.problem_evaluating_strategy

        if self.task_evaluating_strategy:
            if self.tp_modeler is not None:
                task_truth_stream = self.tp_modeler.reconstruct_datastream(
                    problem_inference_info.truth_stream, scenario=TMScenario.Evaluation,
                    with_truth=True
                )
                task_inference_stream = self.tp_modeler.reconstruct_datastream(
                    problem_inference_info.inferenced_stream, scenario=TMScenario.Evaluation,
                    with_truth=False
                )
            else:
                task_truth_stream = problem_inference_info.truth_stream
                task_truth_stream.level = TMDataLevel.Task
                task_inference_stream = problem_inference_info.inference_stream
                task_inference_stream.level = TMDataLevel.Task

            task_inference_info = EvaluationStreamInfo(
                truth_stream=task_truth_stream,
                inferenced_stream=task_inference_stream,
                local_rank=problem_inference_info.local_rank,
                epoch=problem_inference_info.epoch,
                step=problem_inference_info.step,
                device=problem_inference_info.device
            )

            self.task_evaluating_strategy.on_evaluation_begin()
            self.task_evaluating_strategy.on_stream_inferenced(task_inference_info)

            last_inference_info = task_inference_info
            last_strategy = self.task_evaluating_strategy

        for channel in last_inference_info.truth_stream.eval_channels:
            truth_channel = last_inference_info.truth_stream[channel]
            inference_channel = last_inference_info.inferenced_stream[channel]
            for truth, inference in zip(truth_channel, inference_channel):
                # consume the stream, and actually run the pipeline
                pass

        if self.machine_evaluating_strategy:
            eval_results["machine"] = self.machine_evaluating_strategy.on_evaluation_end(evalation_termination_info)

        if self.problem_evaluating_strategy:
            eval_results["problem"] = self.problem_evaluating_strategy.on_evaluation_end(evalation_termination_info)

        if self.task_evaluating_strategy:
            eval_results["task"] = self.task_evaluating_strategy.on_evaluation_end(evalation_termination_info)

        return eval_results

    @classmethod
    def check_contracts(self):

        validator = TMSystemValidator()
        validator.static_validate(self)

    def build_machine_datastream(self, input_data_stream: TMDataStream):

        inference = not self.is_learning()
        scenario = TMScenario.Learning if not inference else TMScenario.Inference

        if input_data_stream.level == TMDataLevel.Task:

            self.task.register_datastream(input_data_stream)
            self.on_task_ready()
            for callback in self.callbacks:
                callback.on_task_data_loaded(input_data_stream)

            input_data_stream = self.tp_modeler.model_datastream(input_data_stream,
                                                                 scenario=scenario
                                                                )

            for callback in self.callbacks:
                callback.on_problem_data_built(input_data_stream)

        if input_data_stream.level == TMDataLevel.Problem:

            self.problem.register_datastream(input_data_stream)
            self.on_problem_ready()
            for callback in self.callbacks:
                callback.on_problem_data_loaded(input_data_stream)

            if self.pm_modeler is not None:
                input_data_stream = self.pm_modeler.model_datastream(input_data_stream,
                                                                 scenario=scenario)
            else:
                input_data_stream.level = TMDataLevel.Machine

            self.on_operator_ready()
            for callback in self.callbacks:
                callback.on_machine_data_built(input_data_stream)

            logger.info(f"machine data build: ")
            for channel in input_data_stream.channels:
                logger.info(f"\t{channel}: {len(input_data_stream[channel])}")

        for callback in self.callbacks:
            callback.on_data_phase_finished(self)

        return input_data_stream

    def recover_input_datastream(self, input_data_stream: TMDataStream):

        inference = not self.is_learning()
        assert input_data_stream.level == TMDataLevel.Machine
        scenario = TMScenario.Learning if not inference else TMScenario.Inference

        if self.pm_modeler is not None:
            input_data_stream = self.pm_modeler.reconstruct_datastream(
                input_data_stream, scenario=scenario, with_truth=not inference)
        elif self.tp_modeler is not None:
            input_data_stream.level = TMDataLevel.Problem

            logger.info(f"problem data recovered: ")
            for channel in input_data_stream.channels:
                logger.info(f"\t{channel}: {len(input_data_stream[channel])}")

        if self.tp_modeler is not None:

            input_data_stream = self.tp_modeler.reconstruct_datastream(
                input_data_stream, scenario=scenario, with_truth=not inference)

        return input_data_stream

    def better_model_discovered(self, info_iter: Union[tuple, SelectedModelInfo]):
        """

        Args:
            info:

        Returns:

        """
        for info in info_iter:
            if isinstance(info, SelectedModelInfo):
                if to_save(self.hyper_params):
                    self.serialize(self.hyper_params.serialize.path)
            elif isinstance(info, tuple):
                metric, machine_info = info
                if to_save(self.hyper_params):
                    self.serialize(self.hyper_params.serialize.multi_path[metric])
            else:
                raise Exception(f"unknown info type: {type(info)}")

    @abc.abstractmethod
    def run(self, source, runtime_options):
        """

            Args:
                source:
                runtime_options:

            Returns:

        """
        pass


    def __getstate__(self):

        return self.__dict__.copy()

    def __setstate__(self, state):

        self.__dict__.update(state)

        if self.operator:

            self.operator.machine = weakref.ref(self.machine)

            self.operator.evaluate_signal.connect(self.evaluate_callback)
            if self.is_learning():
                self.operator.good_model_discovered_signal.connect(
                    self.better_model_discovered)

class TMMultiSystem(TMSystem):
    """
    Multi Task System
    """

    TaskType = TMMultiTask
    TPModelerType = TMMultiModeler
    ProblemType = TMMultiProblem

    PMModelerType = TMMultiModeler
    MachineType = TMMultiMachine

    SubSystems = None

    @classmethod
    def init_class(cls):

        task_types = OrderedDict()
        tp_modeler_types = OrderedDict()
        problem_types = OrderedDict()
        pm_modeler_types = OrderedDict()
        machine_types = OrderedDict()


        for task, system in cls.SubSystems.items():
            task_types[task] = system.TaskType
            tp_modeler_types[task] = system.TPModelerType
            problem_types[task] = system.ProblemType
            pm_modeler_types[task] = system.PMModelerType
            machine_types[task] = system.MachineType

        if all(v is None for v in task_types.values()):
            ThisTaskType = None
        else:
            class ThisTaskType(cls.TaskType):
                SubComponents = task_types
            ThisTaskType.init_class()
        cls.TaskType = ThisTaskType

        task2modeler_contract = dict()
        modeler2problem_contract = dict()
        for k in cls.SubSystems.keys():
            task2modeler_contract[k] = cls.SubSystems[k].Task2ModelerContract
            modeler2problem_contract[k] = cls.SubSystems[k].Modeler2ProblemContract
            if tp_modeler_types[k] is None and cls.SubSystems[k].Task2ProblemContract is not None:
                task2modeler_contract[k] = cls.SubSystems[k].Task2ProblemContract
                tp_modeler_types[k] = TMContractOnlyModeler

        if all(v is None for v in tp_modeler_types.values()):
            ThisTPModeler = None
            cls.Task2ProblemContract = None
        else:
            class ThisTPModeler(cls.TPModelerType):
                SubComponents = tp_modeler_types
            ThisTPModeler.init_class()

            cls.Task2ModelerContract = TMMultiContract(task2modeler_contract)
            cls.Modeler2ProblemContract = TMMultiContract(modeler2problem_contract)

        cls.TPModelerType = ThisTPModeler

        if all(v is None for v in problem_types.values()):
            ThisProblem = None
        else:
            class ThisProblem(cls.ProblemType):
                SubComponents = problem_types
            ThisProblem.init_class()

        cls.ProblemType = ThisProblem

        problem2modeler_contract = dict()
        modeler2machine_contract = dict()
        for k in cls.SubSystems.keys():
            problem2modeler_contract[k] = cls.SubSystems[k].Problem2ModelerContract
            modeler2machine_contract[k] = cls.SubSystems[k].Modeler2MachineContract
            if pm_modeler_types[k] is None and cls.SubSystems[k].Problem2MachineContract is not None:
                problem2modeler_contract[k] = cls.SubSystems[k].Problem2MachineContract
                pm_modeler_types[k] = TMContractOnlyModeler

        if all(v is None for v in pm_modeler_types.values()):
            ThisPMModeler = None
            cls.Problem2MachineContract = None
        else:
            class ThisPMModeler(cls.PMModelerType):
                SubComponents = pm_modeler_types
            ThisPMModeler.init_class()
            cls.Problem2ModelerContract = TMMultiContract(problem2modeler_contract)
            cls.Modeler2MachineContract = TMMultiContract(modeler2machine_contract)

        cls.PMModelerType = ThisPMModeler

        class ThisMachine(cls.MachineType):
            SubComponents = machine_types
        ThisMachine.init_class()
        cls.MachineType = ThisMachine

    def __init__(self, hyper_params,
                 callbacks: Optional[List[TMSystemRuntimeCallbackInterface]] = None,
                 states=None, ):

        self.check_valid_subsystems(self.SubSystems)

        for task, system in self.SubSystems.items():

            if isinstance(hyper_params.subsystems[task], str):
                system_params = self.load_hyper_parameters(hyper_params.subsystems[task])["system"]
            else:
                system_params = hyper_params.subsystems[task]
                assert len(system_params) > 0

            hyper_params.task[task] = TMHyperParams(system_params.task)
            hyper_params.tp_modeler[task] = TMHyperParams(system_params.tp_modeler)
            hyper_params.problem[task] = TMHyperParams(system_params.problem)
            hyper_params.pm_modeler[task] = TMHyperParams(system_params.pm_modeler)
            hyper_params.machine[task] = TMHyperParams(system_params.machine)

            if self.is_learning():
                hyper_params.learner[task] = TMHyperParams(system_params.learner)
            else:
                hyper_params.inferencer[task] = TMHyperParams(system_params.inferencer)
                if hyper_params.inferencer is None:
                    hyper_params.inferencer = TMHyperParams()
                hyper_params.inferencer[task] = TMHyperParams(system_params.system.inferencer)


        super().__init__(hyper_params=hyper_params, callbacks=callbacks, states=states)

    def __getitem__(self, item):

        system_params = TMHyperParams()
        # system_params.task = self.hyper_params.task[item]
        # system_params.tp_modeler = self.hyper_params.tp_modeler[item]
        # system_params.problem = self.hyper_params.problem[item]
        # system_params.pm_modeler = self.hyper_params.pm_modeler[item]
        # system_params.machine = self.hyper_params.machine[item]
        #
        # if self.is_learning():
        #     system_params.learner = self.hyper_params.learner[item]
        # else:
        #     system_params.inferencer = self.hyper_params.inferencer[item]

        for key in self.hyper_params.subsystems[item]:
            # if key not in {"task", "tp_modeler", "problem", "pm_modeler", "machine", "learner", "inferencer"}:
            system_params[key] = self.hyper_params.subsystems[item][key]

        virtual_system = self.SubSystems[item](hyper_params=system_params,
                                     task=self.task[item] if self.task else None,
                                     tp_modeler=self.tp_modeler[item] if self.tp_modeler else None,
                                     problem=self.problem[item] if self.problem else None,
                                     pm_modeler=self.pm_modeler[item] if self.pm_modeler else None,
                                     machine=self.machine[item] if self.machine else None,
                                     operator=self.operator if self.operator else None)


        return virtual_system

    def check_valid_subsystems(self, systems):
        """
        validate subsystems
        """

        # if not isinstance(systems, OrderedDict):
        #     raise TypeError

        for task, system in systems.items():
            if not issubclass(system, TMSystem):
                raise TypeError
    #
    # def update_machine_operator_hyperparams(self, datastream):
    #     for task, system in self.SubSystems.items():
    #         if self.tp_modeler.modelers[task]:
    #             system.update_machine_tpmodeler_hyperparams(self, self.hyper_params.machine[task],
    #                                                         self.tp_modeler.modelers[task])
    @classmethod
    def load_hyper_parameters(self, conf_file_path, cmd_args=None):
        """
        build hyper parameters from config
        """

        from omegaconf import OmegaConf, open_dict
        cmd_args = cmd_args if cmd_args is not None else []
        base_conf = TMConfig.default()
        user_conf = OmegaConf.load(conf_file_path)
        
        cli_conf = OmegaConf.from_cli(cmd_args)

        with open_dict(base_conf), open_dict(user_conf), open_dict(cli_conf):
            conf = OmegaConf.merge(base_conf, user_conf, cli_conf)

        if not conf.job.startup_path:
            conf.job.startup_path = os.getcwd()

        conf = TMHyperParams(OmegaConf.to_container(conf, resolve=True))

#        conf.freeze()

        return conf


    def pre_system_creation(self):
        for key in self.SubSystems:
            self[key].pre_system_creation()

    def post_system_creation(self):
        for key in self.SubSystems:
            self[key].post_system_creation()

    def pre_task_creation(self):
        for key in self.SubSystems:
            self[key].pre_task_creation()

    def post_task_creation(self):
        for key in self.SubSystems:
            self[key].post_task_creation()

    def on_task_ready(self):
        for key in self.SubSystems:
            self[key].on_task_ready()

    def pre_tpmodeler_creation(self):
        for key in self.SubSystems:
            self[key].pre_tpmodeler_creation()

    def post_tpmodeler_creation(self):
        for key in self.SubSystems:
            self[key].post_tpmodeler_creation()

    def pre_problem_creation(self):
        for key in self.SubSystems:
            self[key].pre_problem_creation()

    def post_problem_creation(self):
        for key in self.SubSystems:
            self[key].post_problem_creation()

    def on_problem_ready(self):
        for key in self.SubSystems:
            self[key].on_problem_ready()

    def pre_pmmodeler_creation(self):
        for key in self.SubSystems:
            self[key].pre_pmmodeler_creation()

    def post_pmmodeler_creation(self):
        for key in self.SubSystems:
            self[key].post_pmmodeler_creation()


    def pre_operator_creation(self):
        for key in self.SubSystems:
            self[key].pre_operator_creation()

    def post_operator_creation(self):
        for key in self.SubSystems:
            self[key].post_operator_creation()

    def on_operator_ready(self):
        for key in self.SubSystems:
            self[key].on_operator_ready()

    def pre_machine_creation(self):
        for key in self.SubSystems:
            virtual_system = self[key]
            virtual_system.pre_machine_creation()

    def post_machine_creation(self):
        for key in self.SubSystems:
            self[key].post_machine_creation()


def is_multi_system(system_type):

    assert issubclass(system_type, TMSystem), f"the type {system_type} is not a sub-class of TMSystem"

    if issubclass(system_type, TMMultiSystem):
        return True
    else:
        return False





