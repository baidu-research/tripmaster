from build.lib.TM.core.components.operator.learner import TMLearner
from tripmaster.core.components.operator.strategies.model_selection import BestOneModelSelectionStrategy
from tripmaster.core.components.operator.strategies.optimization import EpochwiseLRUpdateStrategy
from tripmaster.core.components.operator.supervise import TMSuperviseLearner


class TMMultiLearner(TMLearner):
    """
    Multitask TMLearner
    """

    Optimizer = None
    LRScheduler = None
    OptimizationStrategy = EpochwiseLRUpdateStrategy
    ModelSelectionStrategy = BestOneModelSelectionStrategy

    def __init__(self, hyper_params, machine, states=None):

        metric_task = hyper_params.modelselector.chosen
        if metric_task not in hyper_params:
            raise Exception(f"Task {metric_task} not found")

        modelselector_params = hyper_params[metric_task].modelselector

        modelselector_params.metric = metric_task + "." + modelselector_params.metric
        hyper_params.modelselector = modelselector_params

        super().__init__(hyper_params, machine, states=states)

    def create_optimizer(self, machine):
        """
        create optimizers for subtasks
        """

        self.optimizer = dict()
        self.lr_scheduler = dict()

        submodules = machine.submodules()

        if not isinstance(self.Optimizer, dict):
            self.Optimizer = dict((key, self.Optimizer) for key in submodules.keys())
        if not isinstance(self.LRScheduler, dict):
            self.LRScheduler = dict((key, self.LRScheduler) for key in submodules.keys())

        # self.check_optimizer_submodule_match(submodules)

        self.optimizer = dict()
        self.lr_scheduler = dict()

        for name, modules in submodules.items():

            parameters = sum([list(m.parameters()) for m in modules], [])
            if len(parameters) == 0:
                continue

            if name in self.algorithm_params:
                algorithm_params = self.algorithm_params[name]
            else:
                raise Exception(f"cannot find optimizer parameters for module {name}")

            if name in self.lr_scheduler_params:
                lr_scheduler_params = self.lr_scheduler_params[name]
            else:
                raise Exception(f"cannot find lr_scheduler parameters for module {name}")

            if self.Optimizer[name] is None:
                for module in modules:
                    module.requires_grad_(False)

                continue

            self.optimizer[name], self.lr_scheduler[name] = B.OptimizerBehaviors.create_optimization_components(
                parameters,
                self.Optimizer[name], algorithm_params,
                self.LRScheduler[name], lr_scheduler_params,
                self.gradient_clip_val
            )

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
        pass

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


