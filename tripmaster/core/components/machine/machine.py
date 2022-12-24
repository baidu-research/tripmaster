"""
base class for all machines
"""
import abc
from typing import Optional, Type, Mapping, List
import addict
from tripmaster import logging
from tripmaster.core.components.loss import TMSupportLossMixin
from tripmaster.core.components.machine.data_traits import TMSampleBatchTraits, \
    TMSampleMemoryTraits, TMMultiTaskSampleBatchTraits
from tripmaster.core.components.evaluator import TMSupportEvaluatorMixin
from tripmaster.core.concepts.contract import TMContractChannel

from tripmaster.core.concepts.machine import TMMachineInterface

from tripmaster.core.concepts.component import TMSerializableComponent, TMMultiComponentMixin
from tripmaster.core.concepts.scenario import TMScenario

logger = logging.getLogger(__name__)


from tripmaster.core.components.backend import TMBackendFactory

B = TMBackendFactory.get().chosen()
T = B.BasicTensorOperations
M = B.BasicModuleOperations

class TMMachine(M.Module, TMSerializableComponent, TMMachineInterface,
                   TMSupportEvaluatorMixin, TMSupportLossMixin):


    """
    base class for all machines
    """


    DataTraits: Optional[Type[TMSampleMemoryTraits]] = None
    BatchTraits: TMSampleBatchTraits = TMSampleBatchTraits

    def __init__(self, hyper_params, states=None, shared=None):
        M.Module.__init__(self)
        TMSerializableComponent.__init__(self, hyper_params, states=states)
        TMSupportLossMixin.__init__(self, hyper_params.loss)
        TMSupportEvaluatorMixin.__init__(self, hyper_params.evaluator)

        if "arch" in self.hyper_params:
            self.arch_params = self.hyper_params.arch

    @property
    def validate(self):
        return self._validate

    @validate.setter
    def validate(self, validate):
        self._validate = validate

        if self.evaluator:
            self.evaluator.validate = validate

        if self.loss:
            self.loss.validate = validate

    def states(self):

        return {k: T.to_numpy(T.to_device(v, "cpu")) for k, v in self.state_dict().items()}

    def load_states(self, states):
        states = {k: T.to_tensor(v) for k, v in states.items()}
        M.load_state_dict(self, states)

    def submodules(self):

        return {"all": [self]}

    @classmethod
    def attach(cls, contract_graph, role):

        this_component_name = super().attach(contract_graph, role)
        components = [this_component_name]

        for loss_component_name in cls.provide_loss(contract_graph, controller=cls,
                                                                 controller_role=role):
            components.append(loss_component_name)

        for evaluator_component_name in cls.provide_evaluator(contract_graph, controller=cls,
                                                                 controller_role=role):
            components.append(evaluator_component_name)


        contract_graph.add_subsystem("MachineSubSystem", components)

    def forward_with_validation(self, data, scenario):

        if self.validate:
            for key in (TMContractChannel.Source, TMContractChannel.Target):
                assert self.requires(forward=True, channel=key).is_valid(data)

        output = self.forward(data, scenario=scenario)

        if self.validate:
            if scenario == TMScenario.Learning:
                assert self.provides(forward=False, channel=TMContractChannel.Learn).is_valid(output)
                assert self.loss.requires(forward=True,
                                                  channel=TMContractChannel.Truth).is_valid(data)
                assert self.loss.requires(forward=True,
                                                  channel=TMContractChannel.Learn).is_valid(output)
            elif scenario == TMScenario.Evaluation:
                assert self.provides(forward=False, channel=TMContractChannel.Inference).is_valid(output)
                assert self.evaluator.requires(forward=True,
                                                       channel=TMContractChannel.Truth).is_valid(data)
                assert self.evaluator.requires(forward=True,
                                                       channel=TMContractChannel.Inference).is_valid(output)
            elif scenario == TMScenario.Inference:
                assert self.provides(forward=False, channel=TMContractChannel.Inference).is_valid(output)
            else:
                raise NotImplementedError()

        return output 

    @abc.abstractmethod
    def forward(self, input, scenario=None):
        """

        Args:
            input ():
            scenario (None, Str): "learn", "evaluation", "inference"

        Returns:

        """

        raise NotImplementedError()


from tripmaster.core.components.evaluator import TMMultiEvaluator
from tripmaster.core.components.loss import TMMultiLoss

class TMSubModuleMachine(TMMachine):

    Modules: Mapping[str, Type[M.Module]] = None

    def __init__(self, hyper_params, states=None, shared=None):
        super().__init__(hyper_params, states=states, shared=None)

        if shared is not None:
            for key, m in shared.items():
                setattr(self, key, m)

        self.shared = shared


        for key in self.Modules:
            if self.shared is None or key not in self.shared:
                setattr(self, key, self.Modules[key](**self.arch_params[key]))

        if states is not None:
            self.load_states(states)
    #
    # def __getattr__(self, item):
    #     if item in self.modules:
    #         return self.modules[item]
    #     else:
    #         raise AttributeError(f"Attribute {item} is not found")

    def submodules(self):

        return dict((key, [getattr(self, key)]) for key in self.Modules)
    def states(self):
        states = dict()
        for key, module in vars(self).items():
            if isinstance(module, (M.Module, M.Module, M.ModuleList)):
                if self.shared is None or key in self.shared:
                    continue
                states[key] = module.state_dict()

        return states

    def load_states(self, states):

        for key, module in vars(self).items():
            if isinstance(module, (M.Module, M.Module, M.ModuleList)):
                if self.shared is None or key in self.shared:
                    continue

                module.load_state_dict(states[key])


class TMMultiMachine(TMMultiComponentMixin, TMMachine):
    """
    MultiTask Problem
    """

    SubComponents: Mapping[str, Type[TMSubModuleMachine]] = None

    @classmethod
    def init_class(cls):
        super().init_class()
        cls.init_multi_component()

    def __init__(self, hyper_params=None, states=None, shared=None, dependencies=None):
        TMMachine.__init__(self, hyper_params, states)
        TMMultiComponentMixin.__init__(self, hyper_params, states, default_init=False)

        self.shared_modules = shared

        if shared and states:
            for name in self.shared_modules:
                if states[name]:
                    self.shared_modules[name].load_state_dict(states[name])

        self.dependencies = dependencies

        self.sub_components = M.ModuleDict()
        evaluators = dict()
        loss = dict()

        for task, machine_type in self.SubComponents.items():
            self.sub_components[task] = machine_type(hyper_params[task],
                                               states=states[task] if states is not None else None,
                                               shared=self.shared_modules)
            evaluators[task] = self.sub_components[task].evaluator
            loss[task] = self.sub_components[task].loss

        self.evaluator = TMMultiEvaluator(evaluators)

        self.loss = TMMultiLoss(loss, hyper_params.weights)


    def forward(self, inputs, scenario=None):
        """

        Args:
            input ():
            outputs ():

        Returns:

        """
        assert scenario in {"learn", "evaluate", "inference"}

        results = dict()
        for task, machine in self.sub_components.items():
            if self.dependencies is not None:
                for key in self.requires(forward=True, task=task).entries():
                    if key not in inputs[task] and (task, key) in self.dependencies:
                        source_task, source_key = self.dependencies[(task, key)]
                        inputs[task][key] = results[source_task][source_key]

            results[task] = machine.forward(inputs[task], scenario=scenario)

        return results

    def submodules(self):
        modules = dict()
        for task, machine in self.sub_components.items():
            for component, submodule in machine.submodules().items():
                modules[f"{task}.{component}"] = submodule
        return modules

    def states(self):

        return dict((name, module[0].state_dict()) for name, module in self.submodules().items())

    def load_states(self, states):
        # print("loading state in multi machine: ", states.keys())
        for name, module in self.submodules().items():
            module[0].load_state_dict(states[name])


class TMSharedMultiTaskMachine(TMMultiMachine):
    """
    MultiTask Machine with shared components
    """

    SharedModules: List[str] = None
    BatchTraits = TMMultiTaskSampleBatchTraits

    def __init__(self, hyper_params, states=None, dependencies=None):

        shared_module_types = dict((name, set([m.Modules[name] for k, m in self.SubComponents.items()]))
                                   for name in self.SharedModules)

        # make sure all machines has the same shared module types
        assert all(len(shared_module_types[n]) == 1 for n in self.SharedModules)

        shared_module_types = dict((name, list(shared_module_types[name])[0])
                                     for name in self.SharedModules)


        import json
        shared_hyper_params = dict((name, set([json.dumps(hyper_params[key].arch[name], sort_keys=True)
                               for key in self.SubComponents])) for name in self.SharedModules)

        assert all(len(shared_hyper_params[n]) == 1 for n in self.SharedModules)

        shared_hyper_params = dict((name, json.loads(list(shared_hyper_params[name])[0]))
                                  for name in self.SharedModules)

        shared_modules = M.ModuleDict((name, shared_module_types[name](**shared_hyper_params[name]))
                              for name in self.SharedModules)

        super().__init__(hyper_params, states=states, shared=shared_modules, dependencies=dependencies)

    @abc.abstractmethod
    def shared(self, inputs):
        """

        Args:
            inputs:

        Returns:

        """
        raise NotImplementedError()

    @abc.abstractmethod
    def private_forward(self, task, scenario, shared_output, inputs):
        """

        Args:
            task:
            shared_output:
            inputs:

        Returns:

        """
        raise NotImplementedError()


    def forward(self, inputs, scenario=None):
        """

        Args:
            input ():
            outputs ():

        Returns:

        """
        assert scenario in {"learn", "evaluate", "inference"}

        shared_output = self.shared(inputs)
        results = dict()
        for task, machine in self.sub_components.items():
            if self.dependencies is not None:
                for key in self.requires(forward=True, task=task).entries():
                    if key not in inputs[task] and (task, key) in self.dependencies:
                        source_task, source_key = self.dependencies[(task, key)]
                        inputs[task][key] = results[source_task][source_key]

            results[task] = self.private_forward(task, scenario, shared_output, inputs)

        return results

    def submodules(self):
        modules = addict.Dict()
        for key, m in self.shared_modules.items():
            modules[key] = [m]

        for task, machine in self.sub_components.items():
            for component, submodule in machine.submodules().items():
                if component not in modules:
                    modules[f"{task}.{component}"] = submodule
        return modules

    def states(self):

        return dict((name, module[0].state_dict()) for name, module in self.submodules().items())

    def load_states(self, states):

        for name, module in self.submodules().items():
            module[0].load_state_dict(states[name])

