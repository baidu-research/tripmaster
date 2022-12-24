import inspect

from tripmaster.core.components.evaluator import TMSupportEvaluatorMixin, TMMultiEvaluator
from tripmaster.core.concepts.component import TMSerializableComponent, TMMultiComponentMixin
from tripmaster.core.components.contract import TMContractGraph
from tripmaster import T, P, M
from tripmaster.core.concepts.data import TMDataStream


class TMTask(TMSupportEvaluatorMixin, TMSerializableComponent):
    """
    TMTask
    """

    def __init__(self, hyper_params=None, states=None, default_init=True):
        """

        Args:
            hyper_params:
            states:
            default_init:
        """
        TMSerializableComponent.__init__(self, hyper_params, states=states)
        TMSupportEvaluatorMixin.__init__(self, hyper_params.evaluator, default_init=default_init)

        if states:
            self.load_states(states)


    @property
    def validate(self):
        return self._validate

    @validate.setter
    def validate(self, validate):
        self._validate = validate

        if self.evaluator:
            self.evaluator.validate = validate


    def register_datastream(self, datastream: TMDataStream):

        if not self.validate:
            return

        for channel in list(datastream.learn_channels) + list(datastream.eval_channels):
            for data in datastream[channel]:
                assert self.requires(forward=True).is_valid(data)
                assert self.provides(forward=True).is_valid(data)

        for channel in list(datastream.inference_channels):
            for data in datastream[channel]:
                assert self.requires(forward=True).is_valid(data)

    @classmethod
    def attach(cls, contract_graph: TMContractGraph, role):
        """

        Args:
            role:
            contract_graph:

        Returns:

        """
        this_component_name = super().attach(contract_graph, role)

        components = [this_component_name]

        for evaluator_component_name in super().provide_evaluator(contract_graph, controller=cls,
                                                                 controller_role=role):
            components.append(evaluator_component_name)

        contract_graph.add_subsystem("TaskSubSystem", components)



class TMMultiTask(TMMultiComponentMixin, TMTask):
    

    @classmethod
    def init_class(cls):

        super().init_class()

        cls.init_multi_component()

        cls.Evaluator = None

    def __init__(self, hyper_params=None, states=None, default_init=True):
        TMMultiComponentMixin.__init__(self, hyper_params, states, default_init=default_init)
        TMTask.__init__(self, hyper_params, states, default_init=default_init)


        evaluators = {}
        for name, task in self.sub_components.items():
            if task.evaluator != None:
                evaluators[name] = task.evaluator
                task.evaluator = None

        self.evaluator = TMMultiEvaluator(evaluators)

