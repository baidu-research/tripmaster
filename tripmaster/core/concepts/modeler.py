import abc
from enum import auto
from typing import Dict

from tripmaster.core.concepts.component import TMSerializableComponent, TMMultiComponentMixin
from tripmaster.core.concepts.data import TMDataStream, TMDataChannel, TMDataLevel
from tripmaster.core.concepts.scenario import TMScenario
from tripmaster.utils.enum import AutoNamedEnum

import abc

from tripmaster.core.concepts.component import TMSerializableComponent


class TMDataProcessor(TMSerializableComponent):

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def backward(self, *args, **kwargs):
        pass



class TMModelerInterface(abc.ABC):
    """
    TMProblemModeling
    """
    def update_downstream_component(self, upperstream_component, downstream_component):

        pass

    def model(self, data: Dict, scenario: TMScenario):
        """
        Args:
            data: the data to be modeled
            learn: the data is for learning or not. If is for learning, the modeler can update itself.
        """
        pass


    def reconstruct(self, samples, scenario: TMScenario, with_truth=False):
        """

        Args:
            model ():

        Returns:

        """
        pass

    @abc.abstractmethod
    def model_sample(self, data: Dict, level: TMDataLevel, scenario: TMScenario):
        """
        model the sample by calling model() method, but keep tracking the history using the level info
        Args:
            data: the data to be modeled
            level: the current level of data, for tracking the model history
            scenario: the scenario on which the modeler is running.
        """
        pass

    @abc.abstractmethod
    def reconstruct_sample(self, samples, level: TMDataLevel, scenario: TMScenario):
        """
        reconstruct the sample by calling reconstruct() method, but pop the history using the level info
        Args:
            data: the data to be modeled
            level: the current level of data, for pop the history
            scenario: the scenario on which the modeler is running.
        """
        pass


    @abc.abstractmethod
    def model_datachannel(self, channel: TMDataChannel, scenario: TMScenario):
        pass 

    @abc.abstractmethod
    def reconstruct_datachannel(self, channel: TMDataChannel, scenario: TMScenario):
        pass 

    @abc.abstractmethod
    def model_datastream(self, data_stream: TMDataStream, scenario: TMScenario):
        pass

    @abc.abstractmethod
    def reconstruct_datastream(self, data_stream: TMDataStream, scenario: TMScenario):
        pass


class TMMultiModelerInterface(TMMultiComponentMixin, TMModelerInterface):
    """
    TMMultiModelerInterface
    """

    @classmethod
    def init_class(cls):
        cls.init_multi_component()

        # if default_init:
        #     for task in self.sub_components:
        #         if self[task] is None:
        #             self[task] = TMContractOnlyModeler(None)

    def update_downstream_component(self, upperstream_component, downstream_component):

        for task, modeler in self.sub_components.items():
            if modeler is None:
                continue
            modeler.update_downstream_component(upperstream_component[task], downstream_component[task])

    def set_contract(self, upstream_contract, downstream_contract):
        """

        Args:
            upstream_contract:
            downstream_contract:

        Returns:

        """
        for task, modeler in self.sub_components.items():
            if modeler is None:
                continue
            modeler.set_contract(upstream_contract[task] if upstream_contract else None,
                                 downstream_contract[task] if downstream_contract else None)

