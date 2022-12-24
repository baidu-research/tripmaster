
import abc
import collections
import dataclasses
import os
import pickle
import weakref
from enum import Enum, auto
from typing import Dict, Type

from tripmaster.core.concepts.contract import TMContracted, TMContractChannel
from tripmaster.core.concepts.hyper_params import TMHyperParams
from tripmaster.core.concepts.schema import TMSchema, TMChannelSchema
from tripmaster.utils.enum import AutoNamedEnum

from tripmaster import logging

logger = logging.getLogger(__name__)

class TMConfigurable(object):

    def __init__(self, hyper_params, **kwargs):

        super().__init__()

        if isinstance(hyper_params, TMHyperParams):
            self.hyper_params = hyper_params
        elif dataclasses.is_dataclass(hyper_params):
            self.hyper_params = hyper_params
        elif isinstance(hyper_params, dict):
            self.hyper_params = TMHyperParams(hyper_params)
        elif hyper_params is None:
            self.hyper_params = TMHyperParams()
        else:
            raise Exception(f"unknown hyper_params {hyper_params}")

    def hyper_parameters(self):
        return self.hyper_params

#
# class TMContractChannel(object):
#     """
#     channel define the sub-groups of input.
#     for example, the forward request data may have "source" and "target" sub-groups
#     """
#     ForwardRequests = set()
#     ForwardProvisions = set()
#
#     BackwardRequests = set()
#     BackwardProvisions = set()



#
# class TMContractForwardRequestChannel(TMContractChannel):
#     pass
#
# class TMContractForwardProvisionChannel(TMContractChannel):
#     pass
#
# class TMContractBackwardRequestChannel(TMContractChannel):
#     pass
#
# class TMContractBackwardProvisionChannel(TMContractChannel):
#     pass



class TMComponent(TMConfigurable, TMContracted):
    """
    TMContractedComponent
    """

    ForwardRequestSchema = dict()
    ForwardProvisionSchema = dict()

    BackwardRequestSchema = dict()
    BackwardProvisionSchema = dict()

    @classmethod
    def forward_request_schema(cls):
        return TMChannelSchema(cls.ForwardRequestSchema)

    @classmethod
    def forward_provision_schema(cls):
        return TMChannelSchema(cls.ForwardProvisionSchema)

    @classmethod
    def backward_request_schema(cls):
        return TMChannelSchema(cls.BackwardRequestSchema)

    @classmethod
    def backward_provision_schema(cls):
        return TMChannelSchema(cls.BackwardProvisionSchema)

    @classmethod
    def init_class(cls):

        cls.ForwardRequestSchema = TMChannelSchema(cls.ForwardRequestSchema)
        cls.ForwardProvisionSchema = TMChannelSchema(cls.ForwardProvisionSchema)
        cls.BackwardRequestSchema = TMChannelSchema(cls.BackwardRequestSchema)
        cls.BackwardProvisionSchema = TMChannelSchema(cls.BackwardProvisionSchema)


    def __init__(self, hyper_params, **kwargs):
        super().__init__(hyper_params, **kwargs)

#        type(self).init_class()



    def test(self, config):
        pass


    def requires(self, forward: bool, channel: TMContractChannel=None, *args, **kwargs):
        """

        Args:
            forward:

        Returns:

        """
        if forward:
            schema = self.forward_request_schema()[channel] if channel else \
                     self.forward_request_schema().all()
        else:
            schema = self.backward_request_schema()[channel] if channel else \
                     self.backward_request_schema().all()

        return schema

    def provides(self, forward: bool, channel: TMContractChannel=None, *args, **kwargs):
        """

        Args:
            forward:

        Returns:

        """
        if forward:
            schema = self.forward_provision_schema()[channel] if channel else \
                     self.forward_provision_schema().all()
        else:
            schema = self.backward_provision_schema()[channel] if channel else \
                     self.backward_provision_schema().all()
        return schema


    @classmethod
    def make_contract_entry(cls, subsystem, forward, request, channel, suffix=None):

        forward = "Forward" if forward else "Backward"
        request = "Requests" if request else "Provision"

        entry = f"{subsystem}.{forward}.{request}.{channel.value}"
        if suffix:
            entry += suffix

        return entry

    @classmethod
    def attach(cls, contract_graph, role):
        """

        Args:
            role:
            contract_graph:

        Returns:

        """

        request_entries = []
        for channel in cls.forward_request_schema().keys():
            entry = cls.make_contract_entry(role, forward=True, request=True,
                                                     channel=channel)
            request_entries.append(entry)
            contract_graph.add_entry(entry,
                                     schema=cls.forward_request_schema()[channel])

        for channel in cls.backward_request_schema().keys():
            entry = cls.make_contract_entry(role, forward=False, request=True,
                                                     channel=channel)
            request_entries.append(entry)

            contract_graph.add_entry(entry,
                                     schema=cls.backward_request_schema()[channel])

        provision_entries = []

        for channel in cls.forward_provision_schema().keys():
            entry = cls.make_contract_entry(role, forward=True, request=False,
                                                     channel=channel)
            provision_entries.append(entry)

            contract_graph.add_entry(entry,
                                     schema=cls.forward_provision_schema()[channel])

        for channel in cls.backward_provision_schema().keys():
            entry = cls.make_contract_entry(role, forward=False, request=False,
                                                     channel=channel)
            provision_entries.append(entry)

            contract_graph.add_entry(entry,
                                     schema=cls.backward_provision_schema()[channel])

        component_name = cls.__name__ + "@" + role
        contract_graph.add_component(component_name,
                                     request_entries=request_entries,
                                     provision_entries=provision_entries)

        return component_name



def merge_hyperparams(user_hyperparams, serialized_params):
    """
    merge two hyperparams, user's has higher priority
    Args:
        user_hyperparams:
        serialized_params:

    Returns:

    """
    import copy
    merged = copy.deepcopy(user_hyperparams)  # user has higher priority
    for key in user_hyperparams:
        if key not in serialized_params:
            logger.warning(f"LOAD WARNING: {key} in user yaml not in ckpt")

    for key in serialized_params:
        if key not in merged:
            logger.warning(f"LOAD WARNING: {key} loaded in serialized ckpt not in user yaml")
            merged[key] = serialized_params[key]
        elif isinstance(user_hyperparams[key], collections.Mapping) and \
            isinstance(serialized_params[key], collections.Mapping):
            merged[key] = merge_hyperparams(user_hyperparams[key], serialized_params[key])
        else:
            if user_hyperparams[key] != serialized_params[key]:
                logger.warning(f"LOAD WARNING: {key} in user yaml: {user_hyperparams[key]} "
                               f"not same with ckpt {serialized_params[key]}")
    return merged


class TMSerializable(TMConfigurable):

    HYPER_PARAMS_KEY = 'hyper_parameters'
    STATES_KEY = "states"

    def __init__(self, hyper_params, states=None, **kwargs):

        super().__init__(hyper_params, states=states, **kwargs)

    def states(self):
        states = dict()
        for name, instance in vars(self).items():
            if isinstance(instance, weakref.ProxyTypes):
                continue
            if isinstance(instance, TMSerializable):
                states[name] = instance.states()

        return states

    def load_states(self, states):

        for name, instance in vars(self).items():
            if isinstance(instance, TMSerializable):
                if name in states:
                    instance.load_states(states[name])

    def secure_hparams(self):
        import addict
        hparams = addict.Dict()
        for name, instance in vars(self).items():

            if isinstance(instance, weakref.ProxyTypes):
                continue
            if isinstance(instance, TMSerializable):
                hparams[name] = instance.secure_hparams()

        for key in self.hyper_params:
            if key not in hparams and key != "serialize":
                hparams[key] = self.hyper_params[key]

        return hparams

    def serialize(self, path):

        states = {self.HYPER_PARAMS_KEY: self.secure_hparams().to_dict(), self.STATES_KEY: self.states()}
        pickle.dump(states, open(path, "wb"))

    @classmethod
    def deserialize(cls, path, hyper_params=None):
        """

        Args:
            path:

        Returns:

        """
        from tripmaster.core.components.repo import TMRepo

        import os
        if not os.path.exists(path):
            logger.info(f"{path} does not exists, try to download from repo")
            try:
                path = TMRepo().get(path)
            except Exception as e:
                logger.info(f"Failed to download from repo")
                raise
        states = pickle.load(open(path, "rb"))
        serialized_hyper_params = TMHyperParams(states[cls.HYPER_PARAMS_KEY]) \
            if cls.HYPER_PARAMS_KEY in states else None
        if hyper_params is None:
            hyper_params = serialized_hyper_params
        else:
            hyper_params = merge_hyperparams(hyper_params, serialized_hyper_params)

        object = cls(hyper_params=hyper_params, states=states[cls.STATES_KEY])
        # object.load_states()
        return object

class TMSerializableComponent(TMComponent, TMSerializable):
    """
    TMSerializableComponent
    """

    def __init__(self, hyper_params, states=None, **kwargs):
        super().__init__(hyper_params, states=states, **kwargs)


    @classmethod
    def create(cls, hyper_param,  *args, **kwargs):
        """

        Returns:

        """
        from tripmaster.core.components.repo import TMRepo

        if not hyper_param:
            return cls(*args, hyper_params=None, **kwargs)

        if hyper_param.serialize and hyper_param.serialize.load:

            serialized_path = os.path.expanduser(hyper_param.serialize.path)
            logger.info(f"trying to load serialized component {serialized_path}")
            if not os.path.exists(serialized_path):

                logger.info(f"{serialized_path} does not exists, try to download from repo")
                repo_uri = serialized_path

                try:
                    serialized_path = TMRepo().get(repo_uri)
                except Exception as e:
                    message = f"Failed to obtain component from repo using uri {repo_uri}"
                    logger.error(message)
                    raise Exception(message)
                hyper_param.serialize.path = serialized_path
            component = cls.deserialize(serialized_path, hyper_param)
            logger.info(f"component loaded from {serialized_path}")
        else:
            component = cls(hyper_param, *args, **kwargs)

        return component
    


class TMMultiComponentMixin(object):
    """
    dynamic component
    """

    SubComponents: Dict[str, Type[TMSerializableComponent]] = None

    @classmethod
    def init_multi_component(cls):

        for key, component in cls.SubComponents.items():
            if component is not None:
                component.init_class()

        cls.ForwardRequestSchema = dict((key, cls.SubComponents[key].ForwardRequestSchema if cls.SubComponents[key] else None)
                                        for key in cls.SubComponents)
        cls.ForwardProvisionSchema = dict((key, cls.SubComponents[key].ForwardProvisionSchema if cls.SubComponents[key] else None)
                                          for key in cls.SubComponents)

        cls.BackwardRequestSchema = dict((key, cls.SubComponents[key].BackwardRequestSchema if cls.SubComponents[key] else None)
                                         for key in cls.SubComponents)
        cls.BackwardProvisionSchema = dict((key, cls.SubComponents[key].BackwardProvisionSchema if cls.SubComponents[key] else None)
                                           for key in cls.SubComponents)

    @classmethod
    def forward_request_schema(cls):
        return TMChannelSchema(dict((key, cls.SubComponents[key].forward_request_schema()
                                    if cls.SubComponents[key] else None)
                                    for key in cls.SubComponents))

    @classmethod
    def forward_provision_schema(cls):
        return TMChannelSchema(dict((key, cls.SubComponents[key].forward_provision_schema()
                                    if cls.SubComponents[key] else None)
                                    for key in cls.SubComponents))

    @classmethod
    def backward_request_schema(cls):
        return TMChannelSchema(dict((key, cls.SubComponents[key].backward_request_schema()
                                    if cls.SubComponents[key] else None)
                                    for key in cls.SubComponents))

    @classmethod
    def backward_provision_schema(cls):
        return TMChannelSchema(dict((key, cls.SubComponents[key].backward_provision_schema()
                                    if cls.SubComponents[key] else None)
                                    for key in cls.SubComponents))
    def __init__(self, hyper_params, states=None, default_init=True):

        self.sub_components = dict()
        if default_init:
            for task, component_type in self.SubComponents.items():
                if component_type is not None:
                    self.sub_components[task] = component_type(hyper_params[task],
                                                       states=states[task] if states is not None else None)
                else:
                    self.sub_components[task] = None


    def __getitem__(self, item):

        return self.sub_components[item]

    def __setitem__(self, key, value):
        self.sub_components[key] = value


    def requires(self, forward: bool, channel: TMContractChannel=None, task=None, *args, **kwargs):

        if task is None:
            schema_dict = dict()
            for task in self.sub_components.keys():
                if self.sub_components[task] is not None:
                    schema_dict[task] = self.sub_components[task].requires(forward, channel, *args, **kwargs).data()
                else:
                    schema_dict[task] = {}

            return TMSchema(schema_dict)
        else:
            return self.sub_components[task].requires(forward, channel, *args, **kwargs)

    def provides(self, forward: bool, channel: TMContractChannel=None, task=None, *args, **kwargs):

        if task is None:
            schema_dict = dict()
            for task in self.sub_components.keys():
                if self.sub_components[task] is not None:
                    schema_dict[task] = self.sub_components[task].provides(forward, channel, *args, **kwargs).data()
                else:
                    schema_dict[task] = {}

            return TMSchema(schema_dict)
        else:
            return self.sub_components[task].provides(forward, channel, *args, **kwargs)


    def states(self):
        states = super().states()

        for task, problem in self.sub_components.items():
            states[task] = problem.states()

        return states

    def load_states(self, states):
        super().load_states(states)
        for task, problem in self.sub_components.items():
            self.sub_components[task].load_states(states[task])

