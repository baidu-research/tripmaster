import inspect

from tripmaster.core.components.evaluator import TMSupportEvaluatorMixin, TMMultiEvaluator
from tripmaster.core.concepts.component import TMSerializableComponent, TMMultiComponentMixin
from tripmaster.core.concepts.data import TMDataStream


class TMProblem(TMSupportEvaluatorMixin, TMSerializableComponent):
    """
    TMProblem
    """

    def __init__(self, hyper_params, states=None, default_init=True):
        """

        Args:
            evaluator:
            loss:
            **kwargs:
        """
        TMSerializableComponent.__init__(self, hyper_params)
        TMSupportEvaluatorMixin.__init__(self, hyper_params.evaluator, default_init=True)

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
    def attach(cls, contract_graph, role):

        if not cls.ForwardRequestSchema:
            cls.ForwardRequestSchema = cls.ForwardProvisionSchema
        if not cls.BackwardProvisionSchema:
            cls.BackwardProvisionSchema = cls.BackwardRequestSchema

        this_component_name = super().attach(contract_graph, role)

        components = [this_component_name]

        for evaluator_component_name in super().provide_evaluator(contract_graph, controller=cls,
                                                                 controller_role=role):
            components.append(evaluator_component_name)

        contract_graph.add_subsystem("ProblemSubSystem", components)


class TMMultiProblem(TMMultiComponentMixin, TMProblem):
    """
    MultiTask Problem
    """


    @classmethod
    def init_class(cls):
        cls.init_multi_component()

        cls.Evaluator = None

    def __init__(self, hyper_params=None, states=None, default_init=True):
        TMProblem.__init__(self, hyper_params, states, default_init=default_init)
        TMMultiComponentMixin.__init__(self, hyper_params, states, default_init=default_init)

        evaluators = {}
        for name, problem in self.sub_components.items():
            if problem.evaluator != None:
                evaluators[name] = problem.evaluator
                problem.evaluator = None

        self.evaluator = TMMultiEvaluator(evaluators)
