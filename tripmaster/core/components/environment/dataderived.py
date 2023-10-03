import itertools
from abc import abstractmethod
from collections import defaultdict
from random import shuffle
from typing import Optional, Union, Dict
import numpy as np
from more_itertools import chunked

from tripmaster.core.components.environment.base import TMEnvironment, \
    TMEnvironmentPool, TMBatchEnvironment, TMEnvironmentPoolGroup
from tripmaster.core.components.modeler.memory_batch import TMMemory2BatchModeler
from tripmaster.core.components.modeler.modeler import TMModeler
from tripmaster.core.concepts.component import TMConfigurable
from tripmaster.core.concepts.data import TMDataStream, TMDataChannel
from tripmaster.core.concepts.scenario import TMScenario


class TMDataDerivedEnvironment(TMEnvironment):
    """
    Environment derived from data, built to complete some data-driven tasks.
    """

    def __init__(self, hyper_params, sample, modeler,
                 scenario=TMScenario.Learning, eval=False, states=None):

        super().__init__(hyper_params=hyper_params,
                         scenario=scenario, eval=eval, states=states)
        self.__sample = sample
        self.__modeler = modeler

    @abstractmethod
    def reset(self, *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,):

        pass

    @abstractmethod
    def accumulated_reward(self, rewards):
        """
        Args:
            rewards:

        Returns:

        """
        pass

    @abstractmethod
    def future_reward(self, rewards):
        """
        Args:
            rewards:

        Returns:

        """
        pass

    @abstractmethod
    def step(self, action):
        """
        observation (object): this will be an element of the environment's :attr:`observation_space`.
                This may, for instance, be a numpy array containing the positions and velocities of certain objects.
            reward (float): The amount of reward returned as a result of taking the action.
            terminated (bool): whether a `terminal state` (as defined under the MDP of the task) is reached.
                In this case further step() calls could return undefined results.
            truncated (bool): whether a truncation condition outside the scope of the MDP is satisfied.
                Typicaly a timelimit, but could also be used to indicate agent physically going out of bounds.
                Can be used to end the episode prematurely before a `terminal state` is reached.
            info (dictionary): `info` contains auxiliary diagnostic information (helpful for debugging, learning, and logging).

        """
        pass



class TMDataDerivedEnvironmentPoolGroup(TMEnvironmentPoolGroup):
    """
    TMDataDerivedEnvironmentPoolGroup
    """

    EnvType: Union[TMEnvironment, Dict[str, TMEnvironment]] = None
    DataStreamType: TMDataStream = None


    def __init__(self, hyper_params,
                 scenario: TMScenario=TMScenario.Learning, eval=False,
                 states=None):
        super().__init__(hyper_params=hyper_params, scenario=scenario,
                         eval=eval, states=states)
        

        if states is not None:
            self.load_states(states)
            return 

        env_type_dict = self.EnvType if isinstance(self.EnvType, dict) else defaultdict(lambda: self.EnvType)

        key_set = set(states["pools"].keys()) if isinstance(self.EnvType, dict) else {"$default$"}

        data_stream = self.DataStreamType.create(self.hyper_params.data_stream)

        self.hyper_params.learn_pools = data_stream.learn_channels
        self.hyper_params.eval_pools = data_stream.eval_channels
        self.hyper_params.inference_pools = data_stream.inference_channels

        if scenario == TMScenario.Learning:
            pool_names = data_stream.learn_channels
        else:
            pool_names = data_stream.inference_channels
        
        for channel in pool_names + self.eval_pools:
            key = channel.split('.')[0]
            env_type = env_type_dict[key]
            if channel in self.hyper_params.pools:
                params_key = channel
            else:
                params_key = "$default$"

            env_hyper_params = self.hyper_params.pools[params_key].env

            eval = channel in self.eval_pools

            used_data = data_stream[channel]
            if self._test_config is not None:
                used_data = list(itertools.islice(used_data, self._test_config.sample_num))

            self.add_pool(env_type=env_type,
                                env_hyper_params=env_hyper_params,
                                data_channel=TMDataChannel(data=used_data, name=channel, level=data_stream.level),
                                scenario=scenario,
                                eval=eval)

    def add_pool(self, env_type, env_hyper_params, data_channel: TMDataChannel,
                    scenario: TMScenario, eval: bool = False):

        envs = [env_type(hyper_params=env_hyper_params,
                         sample=sample,
                         scenario=TMScenario.Learning, eval=eval)
                for sample in data_channel]


        self[data_channel.name] = TMEnvironmentPool(hyper_params=None,
                                                         name=data_channel.name, envs=envs,
                                                         scenario=scenario,
                                                         eval=eval,
                                                         states=None)
        
    
    def states(self):
        for k, v in self._pools.items():
            v.reuse()

        return {"pools": {k: [e.states() for e in v.envs] 
                           for k, v in self._pools.items() if not k.endswith("#sampled")},
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
        
        key_set = set(states["pools"].keys()) if isinstance(self.EnvType, dict) else {"$default$"}

        env_type_dict = self.EnvType if isinstance(self.EnvType, dict) else defaultdict(lambda: self.EnvType)
        self._pools = dict()
        for k, v in states["pools"].items():
            if k.endswith("#sampled"):
                continue

            envs = [env_type_dict[k](hyper_params=self.hyper_params.pools[k].env, states=states)
                    for states in v]
            
            self._pools[k] = TMEnvironmentPool(hyper_params=None, name=k,
                                                    envs=envs, scenario=self.scenario, eval=self._eval)
        

