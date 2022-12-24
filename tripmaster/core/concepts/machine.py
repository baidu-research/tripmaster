"""
machine interface
"""
import abc
from typing import Optional, Type




class TMMachineInterface(object):


    @abc.abstractmethod
    def submodules(self):
        """
        return a dict from name to submodule list
        Returns:

        """
        pass

    @abc.abstractmethod
    def forward(self, input, scenario=None):
        raise NotImplementedError()



