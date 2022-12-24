"""
loss interface
"""
import abc
from enum import auto

from tripmaster.core.concepts.contract import TMContracted


class TMLossInterface(TMContracted):

    @abc.abstractmethod
    def __call__(self, machine_output, target):
        """

        Args:
            pred ():
            target ():

        Returns:

        """
        raise NotImplementedError()

#
#
# class TMLossContractRequireChannel(TMContractRequireChannel):
#
#     TRUTH = auto()
#     LEARN = auto()
