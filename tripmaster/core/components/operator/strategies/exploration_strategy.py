from abc import abstractmethod

from tripmaster.core.components.environment.base import TMBatchEnvironmentInterface
from tripmaster.core.components.machine.machine import TMMachine
from tripmaster.core.components.machine.reinforce import TMPolicyMachineInterface
from tripmaster.core.concepts.component import TMConfigurable
from tripmaster import P, T, M, D
import numpy as np

from tripmaster.utils.sample import batchwise_select


class TMExplorationStrategy(TMConfigurable):
    """
    TMExplorationStrategy

    """

    @abstractmethod
    def explore(self, machine, batch_env, device):

        raise NotImplementedError()


class TMMAPPolicyExploreStrategy(TMExplorationStrategy):
    """
    PangMapPolicyExploreStrategy: generate trajectories according to action = argmax_a(pi(a|s))
    """

    def __init__(self, hyper_params):
        super().__init__(hyper_params)
        self.hyper_params = hyper_params

    def explore(self, machine: TMPolicyMachineInterface, batch_env: TMBatchEnvironmentInterface, device):
        """
        explore trajectories from the environment
        """
        observations = batch_env.reset(return_info=False)

        batch_mask = [obs is None for obs in observations]
        finished = batch_mask[:]
        trajectories = []
        observation_batch = machine.batch_traits.batch(observations)
        last_observations = observation_batch
        while not all(finished):
            observation_batch = machine.batch_traits.to_device(observation_batch, device)
            action_batch = machine.policy(observation_batch, batch_mask=finished)
            actions = machine.batch_traits.unbatch(action_batch)
            observations, rewards, truncated, terminated, info = batch_env.step(actions, batch_mask=finished)
            observation_batch = machine.batch_traits.batch(observations)
            action_batch = machine.batch_traits.batch(actions)
            reward_batch = np.array(rewards)
            finished_batch = machine.batch_traits.batch(terminated)
            step_data = {"observation": last_observations, "action": action_batch, "reward": reward_batch,
                         "result_observation": observation_batch, "info": info, "batch_mask": finished_batch}
            trajectories.append(step_data)

            finished = [f or tru or ter for f, tru, ter in zip(finished, truncated, terminated)]

        batched_all_rewards = np.stack([step_data["reward"] for step_data in trajectories])
        future_rewards = np.stack(batch_env.future_reward(batched_all_rewards))

        for idx, step_data in enumerate(trajectories):
            step_data["future_reward"] = future_rewards[:, idx]

        return trajectories

class TMGreedyExplorationStrategy(TMExplorationStrategy):
    """
    TMGreedyExploration
    """

    def explore(self, machine: TMMachine, batch_env, device):

        observations = batch_env.reset(return_info=False)

        batch_size = machine.BatchTraits.batch_size(observations)

        batch_mask = [False] * batch_size
        finished = batch_mask[:]
        trajectories = []
        while not all(finished):
            observations = machine.BatchTraits.to_device(observations, device)
            explore_result = machine.explore(observations, batch_mask=finished)

            explored_actions = explore_result["actions"]  # shape = (batch_size, exploration_attempts, action_dim...)
            log_prob = explore_result["log_prob"]
            max_prob_index = log_prob.argmax(dim=1)

            actions = batchwise_select(explored_actions, index=max_prob_index)

            batch_mask = [f or masked for f, masked in zip(finished, batch_mask)]
            new_observations, rewards, truncated, terminated, info = batch_env.step(actions)

            finished = [f or tru or ter for f, tru, ter in zip(finished, truncated, terminated)]

            step_data = {"observations": observations, "actions": actions, "rewards": rewards,
                         "new_observations": new_observations,
                         "terminated": np.array(finished),
                         "batch_mask": np.array(batch_mask), "info": info}
            trajectories.append(step_data)

        batched_all_rewards = np.stack([step_data["rewards"] for step_data in trajectories])

        future_rewards = np.stack(batch_env.future_reward(batched_all_rewards))


        for idx, step_data in enumerate(trajectories):
            step_data["future_rewards"] = future_rewards[idx, :]

        return trajectories