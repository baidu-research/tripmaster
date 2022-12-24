"""
enviornments
"""
from abc import abstractmethod


class TMPolicyMachineInterface(object):
    """
    TMPolicyMachine
    """

    @abstractmethod
    def explore(self, observation, batch_mask=None):
        """
        explore action from action space according to the observation
        Args:
            observation:
        Returns:
            {actions: a, log_prob: log(p(a|s))}

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



