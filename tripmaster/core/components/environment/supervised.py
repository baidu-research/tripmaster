from abc import abstractmethod
from typing import Optional
import numpy as np
from tripmaster.core.components.environment.base import TMEnvironment, TMBatchEnvironmentInterface, \
    TMEnvironmentPool
from tripmaster.core.components.modeler.memory_batch import TMMemory2BatchModeler
from tripmaster.core.components.modeler.modeler import TMModeler
from tripmaster.core.concepts.data import TMDataStream
from tripmaster.core.concepts.scenario import TMScenario


class TMSupervisedDataEnvironment(TMEnvironment):
    """
    TMSupervisedDataEnvironment
    """

    def __init__(self, hyper_params, sample, level):

        super().__init__(hyper_params=hyper_params)
        self.__sample = sample
        self.__level = level

    def level(self):
        return self.__level

    def reset(self, *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,):

        return self.__sample

    def accumulated_reward(self, rewards):
        """
        Args:
            rewards:

        Returns:

        """
        return rewards[0]

    def future_reward(self, rewards):
        """
        Args:
            rewards:

        Returns:

        """
        return rewards

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
        observ = self.__sample
        reward = 1.0
        terminated = True
        truncated = False
        info = {}
        return observ, reward, terminated, truncated, info

    def truth(self):

        return self.__sample

class TMSupervisedDataBatchEnvironment(TMSupervisedDataEnvironment,
                                          TMBatchEnvironmentInterface):
    """
    TMSupervisedDataEnvironment
    """

    def __init__(self, hyper_params, sample, level, batch_size):

        super().__init__(hyper_params=hyper_params, sample=sample, level=level)
        self.__batch_size = batch_size

    def batch_size(self):
        return self.__batch_size

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
        observ = action
        reward = np.ones(self.batch_size())
        terminated = np.ones(self.batch_size(), dtype=np.bool)
        truncated = np.zeros(self.batch_size(), dtype=np.bool)
        info = {}
        return observ, reward, terminated, truncated, info

class TMSupervisedDataEnvironmentPool(TMEnvironmentPool):
    """
    TMSupervisedDataEnvironmentPool
    """

    EnvironmentType = TMSupervisedDataEnvironment
    BatchEnvironmentType = TMSupervisedDataBatchEnvironment

    def __init__(self, hyper_params, datastream: TMDataStream = None, states=None):

        self.__datastream = datastream
        self.__scenario = None
        super().__init__(hyper_params, states=states, level=datastream.level)

    @property
    def datastream(self):
        return self.__datastream

    def states(self):

        return {"datastream": self.__datastream.states()}

    def load_states(self, states):

        self.__datastream = TMDataStream(None, states=states["datastream"])
        self.__level = self.__datastream.level

    def apply_modeler(self, modeler: TMModeler, scenario: TMScenario):

        datastream = modeler.model_datastream(self.__datastream, scenario)

        env_pool = self.__class__(hyper_params=None, datastream=datastream)
        env_pool.scenario = scenario
        return env_pool

    def batchify(self, batch_modeler: TMMemory2BatchModeler, scenario: TMScenario):

        batch_data_stream = batch_modeler.model_datastream(self.__datastream, scenario)
        class BatchEnvPoolType(TMSupervisedDataBatchEnvironmentPool):
            BatchEnvironmentType = self.BatchEnvironmentType

        batch_env_pool = BatchEnvPoolType(hyper_params=None,
                                datastream=batch_data_stream,
                                batch_traits=batch_modeler.batch_traits)
        batch_env_pool.scenario = scenario

        return batch_env_pool
    #
    # @property
    # def scenario(self):
    #     return self.__scenario
    #
    # @scenario.setter
    # def scenario(self, value: TMScenario):
    #     self.__scenario = value
    #
    #

    def envs(self):

        if self.scenario == TMScenario.Learning:
            channels = list(self.__datastream.learn_channels)
        elif self.scenario == TMScenario.Evaluation:
            channels = list(self.__datastream.eval_channels)
        elif self.scenario == TMScenario.Inference:
            channels = list(self.__datastream.inference_channels)
        else:
            raise Exception(f"Unknown scenario value {self.scenario}")

        for channel in channels:
            for sample in self.__datastream[channel]:
                yield self.EnvironmentType(None, sample, level=self.level)

    def test(self, test_config):
        self.__datastream.test(test_config)


class TMSupervisedDataBatchEnvironmentPool(TMSupervisedDataEnvironmentPool):

    BatchEnvironmentType = TMSupervisedDataBatchEnvironment
    def __init__(self, hyper_params, datastream: TMDataStream,
                 batch_traits, states=None):
        super().__init__(hyper_params, datastream, states)
        self.batch_traits = batch_traits

    def apply_modeler(self, modeler: TMModeler, scenario: TMScenario):
        raise RuntimeError(f"Should never be called")

    def batchify(self, batch_modeler, scenario: TMScenario):
        raise RuntimeError(f"Should never be called")

    def envs(self):

        if self.scenario == TMScenario.Learning:
            channels = list(self.datastream.learn_channels)
        elif self.scenario == TMScenario.Evaluation:
            channels = list(self.datastream.eval_channels)
        elif self.scenario == TMScenario.Inference:
            channels = list(self.datastream.inference_channels)
        else:
            raise Exception(f"Unknown scenario value {self.scenario}")

        for channel in channels:
            for sample in self.datastream[channel]:

                yield self.BatchEnvironmentType(None, sample, level=self.datastream.level,
                                                batch_size=self.batch_traits.batch_size(sample))

class TMDefaultSupervisedDataEnvironmentPool(TMSupervisedDataEnvironmentPool):

    DataStreamType = None

    def __init__(self, hyper_params, datastream: TMDataStream = None, states=None):
        if datastream is None:
            datastream = self.DataStreamType(hyper_params.datastream)
        super().__init__(hyper_params, datastream=datastream, states=states)