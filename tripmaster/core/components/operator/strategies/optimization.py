from collections.abc import Sequence, Mapping
from typing import Type
import numpy as np

from tripmaster.core.concepts.component import TMSerializableComponent
from tripmaster.core.components.backend import TMBackendFactory

B = TMBackendFactory.get().chosen()
T = B.BasicTensorOperations
M = B.BasicModuleOperations
O = B.OptimizerBehaviors
class TMOptimizationStrategy(TMSerializableComponent):

    Name: str = None

    def __init__(self, hyper_params, optimizer, lr_scheduler, gradient_clip_val):
        super().__init__(hyper_params)

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.gradient_clip_val = gradient_clip_val

    #    @abc.abstractmethod
    def on_epoch_start(self, machine, epoch_num):
        pass

#    @abc.abstractmethod
    def on_epoch_end(self, machine, epoch_num):
        pass

#    @abc.abstractmethod
    def on_batch_start(self, machine, input_data, batch_num):
        pass

#    @abc.abstractmethod
    def on_batch_end(self, machine, learned_data, loss, batch_num):
        pass

    def on_evaluated(self, machine, evaluation_results):
        pass

    def finish(self):
        pass

    def test(self, test_config):
        pass

    def states(self):
        """

        Returns:
        np_s = {k: {kk: vv.cpu().numpy() if isinstance(vv, torch.Tensor) else vv
        for kk, vv in v.items()} for k, v in state.items()}
        """
        if isinstance(self.optimizer, Sequence):  # todo change to Mapping
            optimizer_states = [{k: T.to_numpy(T.to_device(v, "cpu"))
                                 for k, v in optim.state_dict().items()} for optim in self.optimizer]
        elif isinstance(self.optimizer, Mapping):
            optimizer_states = {}
            for name, optim in self.optimizer.items():
                optimizer_states[name] = O.optimizer_state_dict(optim)
            # optimizer_states = dict((name, {k: T.to_numpy(T.to_device(v, "cpu"))
            #                                 for k, v in optim.state_dict().items()})
            #                         for name, optim in self.optimizer.items())
        else: # todo change to Mapping
            optimizer_states = {k: T.to_numpy(T.to_device(v, "cpu")) for k, v in self.optimizer.state_dict().items()}
            
        if isinstance(self.lr_scheduler, Sequence):
            lr_scheduler_states = [optim.state_dict() for optim in self.lr_scheduler]
        elif isinstance(self.lr_scheduler, Mapping):
            lr_scheduler_states = dict((name, optim.state_dict()) for name, optim in self.lr_scheduler.items())
        else:
            lr_scheduler_states = self.lr_scheduler.state_dict()
            
        return {"optimizer": optimizer_states, "lr_scheduler": lr_scheduler_states,
                "gradient_clip_val": self.gradient_clip_val}

    def load_states(self, states):
        """

        Args:
            states:

        Returns:

        """
        optimizer_states = states["optimizer"]
        if isinstance(self.optimizer, Sequence):
            for optim, states in zip(self.optimizer, optimizer_states):
                M.load_state_dict(optim, states)
        elif isinstance(self.optimizer, Mapping):
            for name, optim in self.optimizer.items():
                O.load_optimizer_state(optim, optimizer_states[name])
        else:
            M.load_state_dict(self.optimizer, optimizer_states)
            
        lr_scheduler_states = states["lr_scheduler"]
        if isinstance(self.lr_scheduler, Sequence):
            for optim, states in zip(self.lr_scheduler, lr_scheduler_states):
                M.load_state_dict(optim, states)
        elif isinstance(self.lr_scheduler, Mapping):
            for name, optim in self.lr_scheduler.items():
                M.load_state_dict(optim, lr_scheduler_states[name])
        else:
            M.load_state_dict(self.lr_scheduler, lr_scheduler_states)
            
        self.gradient_clip_val = states["gradient_clip_val"]

class StepWiseLRUpdateStrategy(TMOptimizationStrategy):

    Name = "stepwise_lrupdate"

    def __init__(self, hyper_params, *args, **kwargs):
        super().__init__(hyper_params, *args, **kwargs)

        self.epochs = hyper_params.epochs
        self.cur_epoch = None

    def on_epoch_start(self, machine, epoch_num):

        self.cur_epoch = epoch_num

    def on_epoch_end(self, machine,  epoch_num):

        self.cur_epoch = epoch_num

    #    @abc.abstractmethod
    def on_batch_start(self, machine,  input_data, batch_num):

        B.OptimizerBehaviors.clear_grad(self.optimizer)
        
    #    @abc.abstractmethod
    def on_batch_end(self, machine, learned_data, loss, batch_num):

        loss.backward()
        # model_params = []
        # for group in self.optimizer['all'].param_groups:
        #     for p in group['params']:
        #         if p.grad is not None:
        #             model_params.append(p)
        # gnorm = B.nn.utils.clip_grad_norm_(model_params, self.gradient_clip_val)
        B.OptimizerBehaviors.clip_gradient_norm(machine, self.gradient_clip_val)
        B.OptimizerBehaviors.optimizer_step(self.optimizer)
        B.OptimizerBehaviors.lrscheduler_step(self.lr_scheduler)


    def finish(self):
        return self.cur_epoch >= self.epochs - 1

    def test(self, test_config):
        if test_config.epoch_num:
            self.epochs = test_config.epoch_num

class EpochwiseLRUpdateStrategy(TMOptimizationStrategy):

    Name = "epochwise_lrupdate"

    def __init__(self, hyper_params, *args, **kwargs):
        super().__init__(hyper_params, *args, **kwargs)

        self.epochs = hyper_params.epochs
        self.cur_epoch = None

    def on_epoch_start(self, machine, epoch_num):

        self.cur_epoch = epoch_num

    def on_epoch_end(self, machine, epoch_num):

        B.OptimizerBehaviors.lrscheduler_step(self.lr_scheduler)
        self.cur_epoch = epoch_num

    #    @abc.abstractmethod
    def on_batch_start(self, machine, input_data, batch_num):

        B.OptimizerBehaviors.clear_grad(self.optimizer)

    #    @abc.abstractmethod
    def on_batch_end(self, machine, learned_data, loss, batch_num):

        loss.backward()
        # model_params = []
        # for group in self.optimizer['all'].param_groups:
        #     for p in group['params']:
        #         if p.grad is not None:
        #             model_params.append(p)
        # gnorm = B.nn.utils.clip_grad_norm_(model_params, self.gradient_clip_val)

        B.OptimizerBehaviors.clip_gradient_norm(machine, self.gradient_clip_val)
        B.OptimizerBehaviors.optimizer_step(self.optimizer)

    def finish(self):
        return self.cur_epoch >= self.epochs

    def test(self, test_config):
        if test_config.epoch_num:
            self.epochs = test_config.epoch_num


class TMOptimizationStrategyFactory(object):

    instance = None

    def __init__(self):

        self.name_strategy_map = dict()

    def register(self, strategy: Type[TMOptimizationStrategy]):

        self.name_strategy_map[strategy.Name] = strategy

    def strategies(self):

        for name in self.name_strategy_map.keys():
            yield name

    def choose(self, name) -> TMOptimizationStrategy:

        return self.name_strategy_map[name]

    @classmethod
    def get(cls):
        """

        Returns:

        """
        if cls.instance is None:
            cls.instance = TMOptimizationStrategyFactory()

        return cls.instance
