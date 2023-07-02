"""
enviornments
"""
from abc import abstractmethod

from tripmaster.core.components.machine.data_traits import TMSampleBatchTraits


class TMPolicyMachineInterface(object):
    """
    TMPolicyMachine
    """

    ObservationBatchTraits = TMSampleBatchTraits
    ActionBatchTraits = TMSampleBatchTraits

    @abstractmethod
    def action_distribution(self, observation, batch_mask=None):
        """
        compute the distribution of actions according to the observation
        Args:
            observation:
        Returns:
            [batch_size, action_numbers, action_space_size]

        """
        raise NotImplementedError()

    @abstractmethod
    def policy(self, observation, batch_mask=None):
        """
        the policy function that returns the action to be taken according to the observation
        Args:
            observation:

        Returns:

        """
        pass

    @abstractmethod
    def action_prob(self, observation, action, batch_mask=None):
        """
        return the ``log prob'' of the action given the observation
        Args:
            observation:
            action:

        Returns:
            log(pi(a|s))
        """
        pass



