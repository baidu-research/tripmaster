"""
base class for TM loss
"""
import abc
import copy
import inspect
import itertools
from enum import Enum, auto
from typing import Union, Dict, Any

from tripmaster.core.concepts.contract import TMContract, TMContractChannel, TMContracted
from tripmaster.core.components.contract import TMContractGraph
from tripmaster.core.concepts.component import TMComponent, TMSerializableComponent

from tripmaster import logging
from tripmaster.core.concepts.loss import TMLossInterface
from tripmaster.core.concepts.schema import TMSchema, TMChannelSchema

logger = logging.getLogger(__name__)

from tripmaster.core.components.backend import TMBackendFactory
B = TMBackendFactory.get().chosen()
M = B.BasicModuleOperations
#
# class TMLossContractEntries(AutoNamedEnum):
#
#     TruthRequire = auto()
#     LearnRequire = auto()


class TMLoss(TMSerializableComponent, TMLossInterface, M.Module):
    """
    TM Loss
    """
    def __init__(self, hyper_params):
        M.Module.__init__(self)
        TMSerializableComponent.__init__(self, hyper_params)
        TMLossInterface.__init__(self)

#        self.hyper_params = hyper_params

    def load_states(self, states):
        pass

    def states(self):

        return {}

class TMFunctionalLoss(TMLoss):

    Func = None
    LearnedFields = None
    TruthFields = None

    def __init__(self, hyper_params=None):
        super().__init__(hyper_params=hyper_params)
    @classmethod
    def forward_request_schema(cls):

        learned_requests = dict((x, object) for x in cls.LearnedFields)
        truth_requests = dict((x, object) for x in cls.TruthFields)

        return TMChannelSchema(
            {TMContractChannel.Truth: truth_requests,
             TMContractChannel.Learn: learned_requests})

    def __call__(self, machine_output, target):
        learned_output = [machine_output[key] for key in self.LearnedFields]
        target_output = [target[key] for key in self.TruthFields]
        func_args = learned_output + target_output

        return self.Func.__func__(*func_args)


class TMContractedLoss(TMLossInterface):
    """
    TMContractAdaptiveLoss
    """
    def __init__(self, loss: TMLoss,
                 truth_contract: Union[TMContract, dict],
                 learn_contract: Union[TMContract, dict]):
        """

        Args:
            loss ():
            machine_adaptor ():
            truth_adaptor ():
        """
        super().__init__()
        self.loss = loss
        self.truth_contract = TMContract(truth_contract) \
            if isinstance(truth_contract, dict) else truth_contract
        self.learn_contract = TMContract(learn_contract) \
            if isinstance(learn_contract, dict) else learn_contract

    def requires(self, forward: bool, channel: TMContractChannel=None, *args, **kwargs):

        if not forward:
            if channel is None:
                return TMChannelSchema({TMContractChannel.Truth: {},
                                       TMContractChannel.Learn: {}})
            else:
                return TMSchema({})
#        assert forward, "loss does not require anything in backward procedure "

        if channel == TMContractChannel.Truth or channel is None:
            truth_schema = self.loss.ForwardRequestSchema[TMContractChannel.Truth]
            if self.truth_contract:
                truth_schema = TMSchema(self.truth_contract.backward(truth_schema.data()))

        if channel == TMContractChannel.Learn or channel is None:
            learn_schema = self.loss.ForwardRequestSchema[TMContractChannel.Learn]
            if self.learn_contract:
                learn_schema = TMSchema(self.learn_contract.backward(learn_schema.data()))

        if channel is None:
            return TMChannelSchema({TMContractChannel.Truth: truth_schema,
                                       TMContractChannel.Learn: learn_schema})
        elif channel == TMContractChannel.Truth:
            return truth_schema
        elif channel == TMContractChannel.Learn:
            return learn_schema
        else:
            raise Exception(f"unsupported channel: {channel}")

    def provides(self, forward: bool, channel: TMContractChannel=None, *args, **kwargs):

        if channel is None:
            return TMChannelSchema({TMContractChannel.Truth: {},
                                       TMContractChannel.Learn: {}})
        else:
            return TMSchema({})

    def __call__(self, machine_output, target):

        adapted_machine_output = self.learn_contract.forward(machine_output) \
            if self.learn_contract else machine_output

        adapted_target = self.truth_contract.forward(target) \
            if self.truth_contract else target
            
        return self.loss(adapted_machine_output,
                         adapted_target)



class TMLossCollection(M.ModuleDict, TMLossInterface):
    """
    TMLossCollection
    """

    def __init__(self, modules: Dict[str, Any], weights: Dict[str, float]):
        """

        Args:
            modules:
        """

        # donot call super().__init__(modules), because it calls the update function
        super().__init__()

        self.non_metric_loss = dict()

        for key, module in modules.items():
            if isinstance(module, M.Module):
                self[key] = module
            else:
                self.non_metric_loss[key] = module

        self.weights = weights

    def requires(self, forward: bool, channel: TMContractChannel=None, *args, **kwargs):

        schema_data = dict()
        for key, x in itertools.chain(self.items(), self.non_metric_loss.items()):
            schema_data.update(x.requires(forward=forward, channel=channel).data())
        return TMSchema(schema_data)

    def provides(self, forward: bool, channel: TMContractChannel=None, *args, **kwargs):

        schema_data = dict()
        for key, x in itertools.chain(self.items(), self.non_metric_loss.items()):
            schema_data.update(x.provides(forward=forward, channel=channel).data())
        return TMSchema(schema_data)

    def __call__(self, machine_output, target):
        return sum(loss(machine_output, target) * self.weights[key]
                      for key, loss in itertools.chain(self.items(), self.non_metric_loss.items()))


class TMMultiLoss(TMLossCollection):
    """
    TMMultiLoss
    """

    def requires(self, forward: bool, channel: TMContractChannel=None, task=None, *args, **kwargs):

        if task is None:
            schema_dict = dict()
            for task, loss in itertools.chain(self.items(), self.non_metric_loss.items()):
                schema_dict[task] = loss.requires(forward, channel, *args, **kwargs).data()

            return TMSchema(schema_dict)
        else:
            loss = self[task] if task in self else self.non_metric_loss[task]
            return loss.requires(forward, channel, *args, **kwargs)

    def provides(self, forward: bool, channel: TMContractChannel=None, task=None, *args, **kwargs):

        if task is None:
            schema_dict = dict()
            for task, loss in itertools.chain(self.items(), self.non_metric_loss.items()):
                schema_dict[task] = loss.provides(forward, channel, *args, **kwargs).data()

            return TMSchema(schema_dict)
        else:
            loss = self[task] if task in self else self.non_metric_loss[task]
            return loss.provides(forward, channel, *args, **kwargs)


    def __call__(self, machine_output, target):

        return sum(self.weights[name] * loss(machine_output[name], target[name])
                   for name, loss in itertools.chain(self.items(), self.non_metric_loss.items()))


class TMSupportLossMixin(object):

    Loss = None

    LossTruthContract = None
    LossLearnContract = None


    @classmethod
    def create_loss(cls, loss, hyper_params, truth_contract, learn_contract):
        if isinstance(loss, TMLossInterface):
            pass
        elif inspect.isclass(loss) and issubclass(loss, TMLossInterface) and\
                issubclass(loss, TMSerializableComponent):
            loss.init_class()
            loss = loss.create(hyper_params)
        else:
            raise Exception(f"unknown evaluator setting {loss}")

        truth_contract = TMContract.parse(truth_contract)
        learn_contract = TMContract.parse(learn_contract)

        if truth_contract or learn_contract:
            loss = TMContractedLoss(loss, truth_contract=truth_contract,
                                                  learn_contract=learn_contract)
        return loss


    def __init__(self, hyper_params):

        self.loss = None

        if self.Loss is None:
            return

        if not isinstance(self.Loss, dict):
            self.loss = self.create_loss(self.Loss, hyper_params,
                                                   self.LossTruthContract,
                                                   self.LossLearnContract)
        else:
            loss = dict()
            for key, evaluator in self.Loss.items():
                loss[key] = self.create_loss(evaluator, hyper_params[key],
                                                        self.LossTruthContract[key],
                                                        self.LossLearnContract[key])
            self.loss = TMLossCollection(loss)

    @classmethod
    def provide_loss(cls, contract_graph: TMContractGraph, controller, controller_role):

        if cls.Loss is None:
            return

        loss_role = f"{controller_role}Loss"

        loss_classes = dict()
        contracts = dict()
        if isinstance(cls.Loss, dict):
            for key, loss in cls.Loss.items():
                role = f"{loss_role}.{key}"
                loss_classes[role] = loss if inspect.isclass(loss) \
                    else loss.__class__
                contracts[role] = (TMContract.parse(cls.LossTruthContract[key]),
                                  TMContract.parse(cls.LossLearnContract[key]))
        else:
            loss_classes[loss_role] = cls.Loss if inspect.isclass(cls.Loss) \
                    else cls.Loss.__class__
            contracts[loss_role] = (TMContract.parse(cls.LossTruthContract),
                                    TMContract.parse(cls.LossLearnContract))

        for role, loss_class in loss_classes.items():

            loss_class.attach(contract_graph, role=loss_role)

            truth_contract, inference_contract = contracts[role]

            channel_mapping = {TMContractChannel.Target: TMContractChannel.Truth,
                               TMContractChannel.Learn: TMContractChannel.Learn}
            contract = {TMContractChannel.Truth: truth_contract,
                        TMContractChannel.Learn: inference_contract}

            contract_graph.connect_consumer(component=controller, component_role=controller_role,
                                              consumer=loss_class,
                                              consumer_role=loss_role,
                                              channel_mapping=channel_mapping,
                                              contract=contract,
                                              )

            yield f"{loss_class.__name__}@{loss_role}"
