"""
TMEnvironment
"""
import enum
import types
from abc import abstractmethod
from itertools import zip_longest
from typing import Type, Optional, Union, Tuple, List, TypeVar

import numpy as np

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")

# try:
#     import gym
#     from gym.core import ObsType, ActType
# except:
#     pass

from tripmaster.core.components.machine.data_traits import TMSampleMemoryTraits, TMSampleBatchTraits
from tripmaster.core.components.modeler.memory_batch import TMMemory2BatchModeler
from tripmaster.core.components.modeler.modeler import TMModeler
from tripmaster.core.concepts.component import TMSerializableComponent, TMConfigurable
from tripmaster.core.concepts.contract import TMContract, TMContractChannel
from tripmaster.core.concepts.data import TMDataChannel, TMDataStream, TMDataLevel
import math

from tripmaster.core.concepts.hyper_params import TMHyperParams
from tripmaster.core.concepts.scenario import TMScenario

class TMEnvironmentInterface:

    @property
    def level(self):
        pass

    @property
    def scenario(self):
        pass

    @scenario.setter
    def scenario(self, value: TMScenario):
        pass

    def accumulated_reward(self, rewards):
        """
        Args:
            rewards:

        Returns:

        """
        pass

    def future_reward(self, rewards):
        """
        Args:
            rewards:

        Returns:

        """
        pass

    def truth(self):
        """
        return the expected truth final observation of the environment
        """
        return None

    @abstractmethod
    def reset(
            self,
            *,
            seed: Optional[int] = None,
            return_info: bool = False,
            options: Optional[dict] = None,
    ) -> Union[ObsType, Tuple[ObsType, dict]]:
        """
        Reset the environment's state.
        Note that this differs from the `reset` method of `gym.Env`, which returns None
        if the random state is exhausted.
        Returns:
            observation (object): the initial observation.

        Args:
            seed:
            return_info:
            options:

        Returns:

        """

        pass

    @abstractmethod
    def step(
        self, action: ActType,
    ) -> Tuple[List[ObsType], List[float], List[bool], List[bool], List[dict]]:

        raise Exception("Not implemented")



class TMEnvironment(TMConfigurable, TMEnvironmentInterface):
    """
    TMEnvironment
    """

    def __init__(self, hyper_params):
        super().__init__(hyper_params=hyper_params)

        if not self.hyper_params.gamma:
            self.hyper_params.gamma = 1.0

    def accumulated_reward(self, rewards):
        """
        Args:
            rewards:

        Returns:

        """
        acc_reward = 0
        for idx, reward in enumerate(rewards):
            if reward is not None:
                acc_reward += math.pow(self.hyper_params.gamma, idx) * reward
        return acc_reward

    def future_reward(self, rewards):
        """
        Args:
            rewards:

        Returns:

        """
        weighted_reward = [math.pow(self.hyper_params.gamma, idx) * reward
                           for idx, reward in enumerate(rewards)]
        future_reward = np.cumsum(weighted_reward[-1::-1])[-1::-1]  # reverse cumsum
        return future_reward


class TMBatchEnvironmentInterface(TMEnvironmentInterface):

    def batch_size(self):

        pass

class TMDefaultBatchEnvironment(TMBatchEnvironmentInterface):
    """
    TMBatchEnvironment
    """

    def __init__(self, hyper_params, env_proto: TMEnvironmentInterface, batch_size):
        super().__init__(hyper_params=hyper_params)
        assert batch_size > 0
        import copy
        self.envs = [copy.copy(env_proto) for _ in batch_size]

    def level(self):
        return self.envs[0].level()

    def batch_size(self):
        return len(self.envs)

    @property
    def scenario(self):
        scenarios = [env.scenario for env in self.envs]
        assert len(set(scenarios)) == 1
        return scenarios[0]

    @scenario.setter
    def scenario(self, value: TMScenario):
        for env in self.envs:
            env.scenario = value

    def __getattr__(self, attr):

        values = [getattr(env, attr) for env in self.envs]
        if isinstance(values[0], (types.FunctionType, types.MethodType)):
            def wrapper(*args, **kwargs):
                results = []
                for idx, func in enumerate(values):
                    result = func(*[arg[idx] for arg in args], **dict((k, v[idx]) for k, v in kwargs.items()))
                    results.append(result)
                return results
            return wrapper
        else:
            return values

    def __setattr__(self, key, value):

        for env in self.envs:
            setattr(env, key, value)

    def reset(
        self,
        *args,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> List[ObsType]:

        return [env.reset(*args, seed, return_info, options)
                for env in self.envs
                ]

    def step(
        self, action: ActType,
    ) -> Tuple[List[ObsType], List[float], List[bool], List[bool], List[dict]]:

        return tuple(zip_longest(*[env.step(act)
                           for env, act in zip_longest(self.envs, action)]))

    def close(self):

        for env in self.envs:
            env.close()


class TMEnvironmentPoolInterface:


    def level(self):
        pass

    @property
    def scenario(self):
        pass

    @scenario.setter
    def scenario(self, value: TMScenario):
        pass

    @abstractmethod
    def apply_modeler(self, modeler, scenario: TMScenario):
        raise NotImplementedError()

    def batchify(self, batch_modeler, scenario: TMScenario):
        raise NotImplementedError()


    @abstractmethod
    def envs(self):
        """
        generate batches of environments
        Args:
            batch_size:
            batch_num:

        Returns:

        """
        raise NotImplementedError

class TMEnvironmentPool(TMSerializableComponent, TMEnvironmentPoolInterface):
    """
    TMEnvironmentPool: preserve a pool of environments for training
    the environments may be identical or different.
    """

    def __init__(self, hyper_params, level, states=None):
        super().__init__(hyper_params, states=states)

        self.__level = level
        self.__scenario = None
        if states:
            self.load_states(states)

    @property
    def level(self):
        return self.__level

    @property
    def scenario(self):
        return self.__scenario

    @scenario.setter
    def scenario(self, value: TMScenario):
        self.__scenario = value

    @abstractmethod
    def test(self, test_config):
        pass

