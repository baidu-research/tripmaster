from collections import defaultdict

from tripmaster.core.components.modeler.modeler import TMModeler, TMMultiModeler
from tripmaster.core.concepts.data import TMDataLevel
from tripmaster.core.concepts.scenario import TMScenario


class TMMachine2MemoryModeler(TMModeler):


    def __init__(self, data_traits, resource_limit):

        super().__init__(None)

        self.data_traits = data_traits
        self.resource_limit = resource_limit


    def clean_reconstructed(self, results, input_level, inference):

        return results


    def model_datachannel(self, data_channel, scenario: TMScenario):
        """

        Args:
            data_channel:
            name:
            with_truth:

        Returns:

        """
        if self.data_traits and self.data_traits.SAMPLE_OOM_POSSIBLE:
            super().model_datachannel(data_channel, scenario)
        else:
            data_channel.level = TMDataLevel.Memory
            return data_channel


    def reconstruct_datachannel(self, channel, scenario: TMScenario, with_truth=False):
        """

        Args:
            channel:
            with_truth:
            inference:

        Returns:

        """
        if self.data_traits and self.data_traits.SAMPLE_OOM_POSSIBLE:
            super().reconstruct_datachannel(channel, scenario)
        else:
            channel.level = TMDataLevel.Machine
            return channel

    def model(self, data, scenario: TMScenario):
        """

        Args:
            inputs ():

        Returns: the modeled data together with the inputs.
            The inputs is included because the reconstruct may need them

        """
        if self.data_traits and self.data_traits.SAMPLE_OOM_POSSIBLE:
            for splitted_data in self.data_traits.fit_memory(data, self.resource_limit):
                if splitted_data is not None:
                    yield splitted_data
        else:
            yield data

    def reconstruct(self, samples, scenario: TMScenario, with_truth=False):
        """

        Args:
            model ():

        Returns:

        """
        if self.data_traits and self.data_traits.SAMPLE_OOM_POSSIBLE:
            return self.data_traits.unfit_memory(samples)
        else:
            assert len(samples) == 1
            return samples[0]


TMMachine2MemoryModeler.init_class()



class TMProtoMultiMachine2MemoryModeler(TMMultiModeler):
    """
    TMMultiMachine2MemoryModeler
    """

    ProtoType = None


    @classmethod
    def init_class(cls):

        cls.SubComponents = defaultdict(lambda: cls.ProtoType)






