"""
contract
"""
import abc
import enum

from bidict import bidict

from tripmaster.core.concepts.schema import TMSchema
from tripmaster import logging
from tripmaster.utils.enum import AutoNamedEnum

logger = logging.getLogger(__name__)


class TMContractInterface(abc.ABC):


    @classmethod
    def parse(cls, input):
        pass

    @classmethod
    def inverse(cls, contract):
        pass

    @abc.abstractmethod
    def forward(self, data):
        pass

    @abc.abstractmethod
    def backward(self, data):
        pass

class TMContract(TMContractInterface):

    def __init__(self, map):

        self.map = bidict(map)

    @classmethod
    def parse(cls, input):

        if input is None:
            return None
        elif isinstance(input, TMContract):
            return input
        elif isinstance(input, dict):
            return TMContract(input)
#        elif issubclass(input, TMKeyMapContract):
#            return input(hyper_params)
        else:
            raise Exception(f"Unknown input for parsing a contract {input}")

    @classmethod
    def inverse(cls, contract):

        if contract is None:
            return None
        else:
            return TMContract(contract.map.inverse)

    def __getitem__(self, item):
        return self.map[item]

    #
    # def requires(self, filter_by_provides=None, *args, **kwargs):
    #     """
    #
    #     Args:
    #         *args ():
    #         **kwargs ():
    #
    #     Returns:
    #
    #     """
    #     if filter_by_provides is None:
    #         return set(self.map.keys())
    #     else:
    #         return set(self.map.inverse[key] if key in self.map.inverse else key
    #                    for key in filter_by_provides)
    #
    # def provides(self, filter_by_requires=None, *args, **kwargs):
    #     """
    #
    #     Args:
    #         *args ():
    #         **kwargs ():
    #
    #     Returns:
    #
    #     """
    #     if filter_by_requires is None:
    #         return set(self.map.inverse.keys())
    #     else:
    #         return set(self.map[key] if key in self.map else key
    #                    for key in filter_by_requires)

    def forward(self, data):
        """

        Args:
            data ():

        Returns:

        """
#        for key in self.map:
#            if self.map[key] in data:
#                message = f"The field {self.map[key]} will be overwritten " \
#                          f"by the contract {self.map}"
#                logger.warning(message)

                # del data[self.map[key]]

        result = dict((k, v) for k, v in data.items())

        for key in self.map:
            result[self.map[key]] = data[key]

        # result = dict((self.map[key], data[key]) if key in self.map else (key, data[key]) for key in data)
        return result

    def backward(self, data):
        """

        Args:
            data ():

        Returns:

        """
        return dict((self.map.inverse[key], data[key]) if key in self.map.inverse else (key, data[key])
                    for key in data)

def parse_contract_adaptor(inputs):
    """
    
    Args:
        inputs: 

    Returns:

    """

    if inputs is None:
        return None
    elif isinstance(inputs, dict):
        if all(isinstance(v, dict) for v in inputs.values()):
            return dict((k, TMContract(v)) for k, v in inputs.items())
        elif all(isinstance(v, str) for v in inputs.values()):
            return TMContract(inputs)
        else:
            raise Exception(f"invalid inputs {inputs} for contract adaptor")
    else:
        raise Exception(f"unknown type of input {type(inputs)} "
                        f"for contract adaptor")


class TMMultiContract(TMContractInterface):

    def __init__(self, contract_entries):

        self.contracts = dict((k, TMContract(v)) for k, v in contract_entries.items())

    @classmethod
    def parse(cls, input):

        if input is None:
            return None
        elif isinstance(input, dict):
            return TMMultiContract(input)
        else:
            raise Exception(f"Unknown input for parsing a contract {input}")

    @classmethod
    def inverse(cls, contract):

        if contract is None:
            return None
        else:
            return TMMultiContract(dict((k, TMContract.inverse(v)) for k, v in contract.contracts))

    def __getitem__(self, item):
        return self.contracts[item]

    def __setitem__(self, key, value):

        self.contracts[key] = TMContract.parse(value)

    def forward(self, data):
        """

        Args:
            data ():

        Returns:

        """
        return dict((k, v.forward(data[k]) if v else data[k]) for k, v in self.contracts)

    def backward(self, data):
        """

        Args:
            data ():

        Returns:

        """
        return dict((k, v.backward(data[k]) if v else data[k]) for k, v in self.contracts)


class TMContractChannel(AutoNamedEnum):

    Source = enum.auto()  # channel for source, used in task and problems
    Target = enum.auto()  # channel for target, used in task and problems

    Learn = enum.auto()   # channel for learning logits, used in loss
    Truth = enum.auto()   # channel for truth, used in evaluator and loss
    Inference = enum.auto() # channel for inferenced results, used in evaluator


class TMContracted(object):
    """
    class declaration is for static checking, while function `requires` and `provides` provide dynamic ability
    """


    def __init__(self, **kwargs):

        self._validate = False

    @property
    def validate(self):
        return self._validate

    @validate.setter
    def validate(self, validate):
        self._validate = validate

    @abc.abstractmethod
    def requires(self, forward: bool, channel: TMContractChannel=None, *args, **kwargs):
        pass

    @abc.abstractmethod
    def provides(self, forward: bool, channel: TMContractChannel=None, *args, **kwargs):
        pass


