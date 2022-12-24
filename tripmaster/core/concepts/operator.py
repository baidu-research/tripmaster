"""
operator
"""
import abc
import weakref
from collections import namedtuple
from typing import Union

from tripmaster.core.components.environment.base import TMEnvironmentPool
from tripmaster.core.concepts.component import TMSerializableComponent
from tripmaster.core.concepts.data import TMDataStream

class TMOperatorInterface(TMSerializableComponent):
    """
    TMMachineOperator
    """

    def __init__(self, hyper_params, machine=None, states=None, **kwargs):
        assert machine is not None

        self.machine = weakref.proxy(machine)

        super().__init__(hyper_params, machine=machine, states=states, **kwargs)



    @abc.abstractmethod
    def operate(self, data_or_env: Union[TMDataStream, TMEnvironmentPool], runtime_options):
        """

        Args:
            machine:
            problem:

        Returns:

        """
        raise NotImplementedError()



