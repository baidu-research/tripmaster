import itertools
from abc import abstractmethod
from collections import defaultdict
from random import shuffle
from typing import Optional, Union, Dict
import numpy as np
from more_itertools import chunked

from tripmaster.core.components.environment.base import TMEnvironment, \
    TMEnvironmentPool, TMBatchEnvironment, TMEnvironmentPoolGroup, TMEnvironmentPoolGroupBuilder
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
    ModelerType: Union[TMModeler, Dict[str, TMModeler]] = None


    def __init__(self, hyper_params,
                 scenario: TMScenario, eval=False,
                 test_config=None,
                 states=None):
        super().__init__(hyper_params=hyper_params, scenario=scenario,
                         eval=eval, test_config=test_config, states=states)


        env_type_dict = self.EnvType if isinstance(self.EnvType, dict) else defaultdict(lambda: self.EnvType)
        modeler_type_dict = self.ModelerType if isinstance(self.ModelerType, dict) else defaultdict(lambda: self.ModelerType)

        data_stream = self.DataStreamType.create(self.hyper_params.data_stream)

        self.learn_pools = data_stream.learn_channels
        self.eval_pools = data_stream.eval_channels
        self.inference_pools = data_stream.inference_channels

        if scenario == TMScenario.Learning:
            pool_names = data_stream.learn_channels
        else:
            pool_names = data_stream.inference_channels

        for channel in pool_names + self.eval_pools:
            key = channel.split('.')[0]
            env_type = env_type_dict[key]
            if channel in self.hyper_params.env:
                params_key = channel
            else:
                params_key = "$default$"

            env_hyper_params = self.hyper_params.env[params_key].config
            modeler_hyper_params = self.hyper_params.env[params_key].modeler

            modeler = modeler_type_dict[key].create(hyper_params=modeler_hyper_params)
            eval = channel in self.eval_pools

            used_data = data_stream[channel]
            if self._test_config is not None:
                used_data = list(itertools.islice(used_data, self._test_config.sample_num))

            self.add_pool(env_type=env_type,
                                env_hyper_params=env_hyper_params,
                                data_channel=TMDataChannel(data=used_data, name=channel, level=data_stream.level),
                                modeler=modeler,
                                scenario=scenario,
                                eval=eval)

    def add_pool(self, env_type, env_hyper_params, data_channel: TMDataChannel, modeler: TMModeler,
                    scenario: TMScenario, eval: bool = False):

        envs = [env_type(hyper_params=env_hyper_params,
                         sample=sample, modeler=modeler,
                         scenario=TMScenario.Learning, eval=False)
                for sample in data_channel]

        self[data_channel.name] = TMEnvironmentPool(hyper_params=None,
                                          name=data_channel.name, envs=envs,
                                          scenario=scenario,
                                          eval=eval,
                                          states=None)

class TMDataDerivedEnvironmentPoolGroupBuilder(TMEnvironmentPoolGroupBuilder):
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

        return self.EnvPoolGroupType.create(self.hyper_params.pool_group,
                                            test_config=self._test_config,
                                            scenario=scenario)



