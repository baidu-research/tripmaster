"""
TMEnvironment
"""
import enum
import types
from abc import abstractmethod
from itertools import zip_longest
from random import shuffle
from typing import Type, Optional, Union, Tuple, List, TypeVar

import numpy as np
from more_itertools import chunked

from tripmaster import T

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


from tripmaster import logging

logger = logging.getLogger()
# try:
#     import gym
#     from gym.core import ObsType, ActType
# except:
#     pass

from tripmaster.core.components.machine.data_traits import TMSampleMemoryTraits, TMSampleBatchTraits
from tripmaster.core.components.modeler.memory_batch import TMMemory2BatchModeler
from tripmaster.core.components.modeler.modeler import TMModeler
from tripmaster.core.concepts.component import TMSerializableComponent, TMConfigurable, TMSerializable
from tripmaster.core.concepts.contract import TMContract, TMContractChannel
from tripmaster.core.concepts.data import TMDataChannel, TMDataStream, TMDataLevel
import math

from tripmaster.core.concepts.hyper_params import TMHyperParams
from tripmaster.core.concepts.scenario import TMScenario


class TMEnvironmentInterface:





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


class TMEnvironment(TMSerializableComponent):
    """
    TMEnvironment
    """

    def __init__(self, hyper_params,
                 scenario=TMScenario.Learning,
                 eval=False,
                 states=None):
        super().__init__(hyper_params=hyper_params,
                         states=states)

        if not self.hyper_params.gamma:
            self.hyper_params.gamma = 1.0

        self._scenario = scenario
        self._eval = eval

    @property
    def scenario(self):
        return self._scenario

    @property
    def eval(self):
        return self._eval
    
    # @property 
    # def device(self):
    #     return self._device

    # @device.setter
    # def device(self, value):
    #     self._device = value

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

    def future_reward(self, explored):
        """
        Args:
            rewards:

        Returns:

        """
        
        weighted_reward = [math.pow(self.hyper_params.gamma, idx) * explore_step["reward"]
                           for idx, explore_step in enumerate(explored)]
        future_reward = np.cumsum(weighted_reward[-1::-1])[-1::-1]  # reverse cumsum
        return future_reward


    @abstractmethod
    def reset(
            self,
            *,
            seed: Optional[int] = None,
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
        """
        Run one timestep of the environment's dynamics.
        Accepts an action and returns a tuple (observation, reward, terminated, truncated, info).
        """

        raise Exception("Not implemented")



class TMBatchEnvironment(TMEnvironment):
    """
    TMBatchEnvironment
    """

    ObservationBatchTraits = TMSampleBatchTraits
    ActionBatchTraits = TMSampleBatchTraits

    def __init__(self, hyper_params=None, envs: TMEnvironmentInterface=None,
                 scenario=TMScenario.Learning, eval=False, states=None):
        super().__init__(hyper_params=hyper_params, scenario=scenario, eval=eval, states=states)

        self.__envs = envs


    def batch_size(self):
        return len(self.__envs)

    # def __getattr__(self, attr):
    #
    #     values = [getattr(env, attr) for env in self.__envs]
    #     if isinstance(values[0], (types.FunctionType, types.MethodType)):
    #         def wrapper(*args, **kwargs):
    #             results = []
    #             for idx, func in enumerate(values):
    #                 result = func(*[arg[idx] for arg in args], **dict((k, v[idx]) for k, v in kwargs.items()))
    #                 results.append(result)
    #             return results
    #         return wrapper
    #     else:
    #         return values
    #
    # def __setattr__(self, key, value):
    #
    #     for env in self.envs:
    #         setattr(env, key, value)

    def reset(
        self,
        *args,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> List[ObsType]:

        results = list(zip(* [env.reset(*args, seed=seed, options=options)
                                    for env in self.__envs
                                    ]))

        observation, info = results
        observation = self.ObservationBatchTraits.batch(list(observation))

        return observation, info


    def step(
        self, action_batch: ActType,
        batch_mask: Optional[Union[List[bool], np.ndarray, T.Tensor]] = None,
    ) -> Tuple[List[ObsType], List[float], List[bool], List[bool], List[dict]]:

        actions = self.ActionBatchTraits.unbatch(action_batch)

        if batch_mask is None:
            batch_mask = [False] * self.batch_size()

        step_results = list(zip(*[env.step(act) if not batch_mask[idx] else [None, 0.0, True, True, {}]
                           for idx, (env, act) in enumerate(zip(self.__envs, actions))]))

        observation, reward, terminated, truncated, info = step_results

        batch_mask = [batch_mask[idx] or terminated[idx] or truncated[idx] for idx in range(self.batch_size())]

        ic(observation)
        observation_batch = self.ObservationBatchTraits.batch(observation)

        reward = T.to_tensor(reward)
        terminated = T.to_tensor(terminated)
        truncated = T.to_tensor(truncated)

        return observation_batch, reward, terminated, truncated, info

    def close(self):

        for env in self.__envs:
            env.close()

    def future_reward(self, explored):
        """
        Args:
            rewards:

        Returns:

        """
        
        weighted_reward = T.stack([math.pow(self.hyper_params.gamma, idx) * explore_step["reward"]
                           for idx, explore_step in enumerate(explored)], dim=-1)

        future_reward = T.flip(T.cumsum(T.flip(weighted_reward, dims=[-1]),dim=-1), dims=[-1])  # reverse cumsum

        return future_reward

class TMEnvironmentPool(TMSerializableComponent):
    """
    TMEnvironmentPool: preserve a pool of environments for training
    the environments may be identical or different.
    """

    def __init__(self, hyper_params, name: str,
                 envs=None,
                 scenario=TMScenario.Learning,
                 eval=False,
                 states=None):
        super().__init__(hyper_params, scenario=scenario, states=states)

        self.__name = name
        self.__scenario = None
        self.__eval = eval
        self.__envs = envs
        if states:
            self.load_states(states)

    def reuse(self):
        pass 

    @property
    def name(self):
        return self.__name

    @property
    def scenario(self):
        return self.__scenario

    @property
    def eval(self):
        return self.__eval

    @property
    def envs(self):
        return self.__envs

    def choose(self, sample_num):

        indexes = list(range(len(self.__envs)))
        shuffle(indexes)

        for index_chunk in chunked(indexes, sample_num):
            yield TMBatchEnvironment(envs=[self.__envs[index] for index in index_chunk])

    @abstractmethod
    def test(self, test_config):
        pass


class TMEnvironmentPoolGroup(TMSerializableComponent):
    """
    TMDataStream
    Some thoughts, but not feasible: "Note: as a fundamental components of TM which across all the data pipeline,
          it should not be subclassed and change its default behavior.
          For old code, please load the data in TMOfflineInputStream and then
          return a TMDataStream"
    """

    def __init__(self, hyper_params=None, scenario=TMScenario.Learning,
                 eval=False,
                 test_config=None,
                 states=None):

        super().__init__(hyper_params)

        self._pools = dict()
        self._scenario = scenario
        self._eval = eval
        self._test_config = test_config

        if states is not None:
            self.load_states(states)
            logger.info("add sampled training eval channel ")

            # if self.hyper_params.train_sample_ratio_for_eval or self.hyper_params.train_sample_ratio_for_eval > 0:
            #     ratio = self.hyper_params.train_sample_ratio_for_eval
            #     self.add_sampled_training_eval_channels(ratio)

            logger.info("sampled training eval channel added")

    def choose(self, sample_num, eval=False):

        if eval:
            pools = self.eval_pools
        else:
            if self.scenario == TMScenario.Learning:
                pools = self.learn_pools
            elif self.scenario == TMScenario.Inference:
                pools = self.inference_pools
            else:
                raise Exception("Unknown scenario with eval=False")

        for pool_name in pools:
            yield from self[pool_name].choose(sample_num=sample_num)


    @property
    def scenario(self):
        return self._scenario

    @property
    def eval(self):
        return self._eval

    @property
    def pools(self):
        return self._pools.keys()


    # def add_sampled_training_eval_channels(self, ratio=None):
    #
    #     if ratio is None:
    #         ratio = self.hyper_params.train_sample_ratio_for_eval
    #         if not ratio or (isinstance(ratio, (int, float)) and ratio < 0):
    #             return
    #
    #     import random, copy
    #     sampled_channels = []
    #     for channel in self.learn_channels:
    #         assert channel in self.__pools
    #
    #         sampled = [copy.deepcopy(sample) for sample in self.__pools[channel]
    #                    if random.random() < ratio]
    #         if len(sampled) <= 0:
    #             continue
    #         sampled_channels.append(f"{channel}#sampled")
    #
    #         self.__pools[f"{channel}#sampled"] = TMDataChannel(data=sampled, level=self.__level)
    #
    #     self.hyper_params.channels.eval = list(set(list(self.hyper_params.channels.eval) + sampled_channels))

    @property
    def learn_pools(self):
        return self.hyper_params.learn_pools if self.hyper_params.learn_pools else []

    @learn_pools.setter
    def learn_pools(self, value):
        self.hyper_params.learn_pools = tuple(value)

    @property
    def eval_pools(self):
        return self.hyper_params.eval_pools if self.hyper_params.eval_pools else []

    @eval_pools.setter
    def eval_pools(self, value):
        self.hyper_params.eval_pools = tuple(value)

    @property
    def inference_pools(self):
        return self.hyper_params.inference_pools if self.hyper_params.inference_pools else []

    @inference_pools.setter
    def inference_pools(self, value):
        self.hyper_params.inference_pools = tuple(value)

    def __getitem__(self, item):

        return self._pools[item]

    def __setitem__(self, key, value):
        if not isinstance(value, TMEnvironmentPool):
            value = TMEnvironmentPool(hyper_params=None, name=key,
                                      envs=value, scenario=self.scenario,
                                      eval=self.eval)

        self._pools[key] = value

    def test(self, test_config):

        for k, v in self._pools.items():
            v.sample_num = test_config.sample_num

        self.add_sampled_learning_eval_pools(ratio=1)

    def states(self):
        for k, v in self._pools.items():
            v.reuse()

        return {"pools": {k: v.envs for k, v in self._pools.items() if not k.endswith("#sampled")},
                "eval": self._eval, "scenario": self._scenario.name}

    def secure_hparams(self):
        import copy
        hyper_params = copy.deepcopy(self.hyper_params)
        for channel in hyper_params.channels:
            hyper_params.channels[channel] = [k for k in hyper_params.channels[channel]
                                              if not k.endswith("#sampled")]
        return hyper_params

    def load_states(self, states):
        self._scenario = TMScenario[states["scenario"]]
        self._eval = states["eval"]
        self._pools = {k: TMEnvironmentPool(hyper_params=None, name=k, 
                                            envs=v, scenario=self.scenario, eval=self._eval)
                        for k, v in states["pools"].items() if not k.endswith("#sampled")}


class TMEnvironmentPoolGroupBuilder(TMSerializableComponent):

    @abstractmethod
    def test(self, test_config):
        pass

    @abstractmethod
    def build(self, ** inputs):
        raise NotImplementedError()


class TMDefaultEnvironmentPoolGroupBuilder(TMEnvironmentPoolGroupBuilder):
    """
    TMDataDerivedEnvironmentPoolGroupBuilder
    """

    EnvPoolGroupType: TMEnvironmentPoolGroup = None

    def __init__(self, hyper_params=None):
        super().__init__(hyper_params=hyper_params)
        self._test_config = None

    def test(self, test_config):
        self._test_config = test_config

    def build(self, scenario: TMScenario):

        if TMSerializableComponent.to_load(self.hyper_params.epg):
            return self.EnvPoolGroupType.deserialize(self.hyper_params.epg.serialize.load)

        return self.EnvPoolGroupType.create(self.hyper_params.epg,  # env pool group
                                            test_config=self._test_config,
                                            scenario=scenario)
