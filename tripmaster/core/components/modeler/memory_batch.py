import logging
from collections import defaultdict

from tripmaster.core.components.modeler.modeler import TMModeler, TMMultiModeler
from tripmaster.core.concepts.data import TMDataChannel, TMDataLevel, TMMultiDataStream
from tripmaster.core.components.operator.strategies.batching import TMBatchingStrategyFactory

from tripmaster.core.components.backend import TMBackendFactory
from tripmaster.core.concepts.scenario import TMScenario

P = TMBackendFactory.get().chosen()


class TMBatchChannel(TMDataChannel):

    def __init__(self, hyper_params=None, data=None, batch_sampler=None):
        super().__init__(hyper_params, data, level=TMDataLevel.Batch)
        self.batch_sampler = batch_sampler


class DataChannelDataset(P.Dataset):

    def __init__(self, data_channel):
        super().__init__()
        self.data_channel = data_channel

    def __getitem__(self, idx):
        return self.data_channel[idx]

    def __len__(self):
        return len(self.data_channel)

class DataChannelIterDataset(P.IterableDataset):

    def __init__(self, data_channel):
        super().__init__()
        self.data_channel = data_channel

    def __iter__(self):
        return self.data_channel.__iter__()

class TMMemory2BatchModeler(TMModeler):
    """
    TMMemory2BatchModeler
    """

    def __init__(self, hyper_params, sample_traits, batch_traits):

        super().__init__(hyper_params)

        batching_strategy_name = self.hyper_params.batching.type
        batching_strategy_class = TMBatchingStrategyFactory().get().choose(batching_strategy_name)
        batching_strategy = batching_strategy_class(self.hyper_params.batching.strategies[batching_strategy_name])

        self.batching_strategy = batching_strategy
        self.batch_traits = batch_traits
        self.sample_traits = sample_traits
        self.for_ddp = self.hyper_params.distributed in {"ddp", "fleet"}


    def model_datachannel(self, data_channel: TMDataChannel, scenario: TMScenario):

        # if name == "learn":
        #     data_channel.degenerate()

        if data_channel.support_random_batch():

            batch_sampler = self.batching_strategy.batch_sampler(data_channel, self.sample_traits,
                                                                 learning=scenario==TMScenario.Learning,
                                                                 distributed=self.for_ddp)

            data_set = DataChannelDataset(data_channel)
            data_loader = P.DataLoader(data_set, batch_sampler=batch_sampler,
                                       num_workers=self.hyper_params.dataloader.worker_num,
                                       #                          pin_memory=self.hyper_params.pin_memory,
                                       timeout=self.hyper_params.dataloader.timeout,
                                       worker_init_fn=None,
                                       collate_fn=self.batch_traits.batch,
                                       )
            return TMBatchChannel(hyper_params=None, data=data_loader, batch_sampler=batch_sampler)
        else:

            assert self.hyper_params.batching.type == "fixed_size"
            batch_size = self.batching_strategy.hyper_params.batch_size

            data_set = DataChannelIterDataset(data_channel)
            data_loader = P.DataLoader(data_set, batch_size=batch_size,
                                       num_workers=self.hyper_params.dataloader.worker_num,
                                       #                          pin_memory=self.hyper_params.pin_memory,
                                       timeout=self.hyper_params.dataloader.timeout,
                                       worker_init_fn=None,
                                       collate_fn=self.batch_traits.batch,
                                       )
            return TMBatchChannel(hyper_params=None, data=data_loader)




    def reconstruct_datachannel(self, channel: TMDataChannel, scenario: TMScenario, with_truth=False):

        for batch in channel:
            for sample in self.batch_traits.unbatch(batch):

                yield sample

TMMemory2BatchModeler.init_class()


class TMDefaultMultiMemory2BatchModeler(TMMultiModeler):
    """
    TMMultiMachine2MemoryModeler
    """

    def __init__(self, proto_modeler):

        self.proto_modeler = proto_modeler
        self.sub_components = defaultdict(lambda: self.proto_modeler)

        super().__init__(None)


class TMMergedMemory2BatchModeler(TMMultiModeler):
    """
    TMMultiMachine2MemoryModeler
    """

    def model_datastream(self, data_stream: TMMultiDataStream, inference=False):
        """
        Merge the data from multiple stream into one target stream, in which the samples
        have an extra layer of key to indicate the source stream.
        Args:
            inputs ():

        Returns: the modeled data together with the inputs.
            The inputs are included because the reconstruct may need them

        """
        # results = []
        # for result_items in zip(modeler.model(data, channel=channel, with_truth=with_truth)
        #     for task, modeler in self.modelers.items()):
        #     result = dict(itertools.chain(*[x.items() for x in result_items]))
        #     results.append(result)

        result = TMMultiDataStream()

        result_level = data_stream.level

        for stream_name in data_stream.streams():
            modeler = self.sub_components[stream_name]
            if modeler is None:
                result[stream_name] = data_stream[stream_name]
            else:
                result_datastream = modeler.model_datastream(data_stream[stream_name], inference=inference)
                result[stream_name] = result_datastream
                result_level = result_datastream.level

        result.level = result_level

        return result

    def reconstruct_datastream(self, data_stream: TMMultiDataStream, inference=False):
        """

        Args:
            model ():

        Returns:

        """
        result = TMMultiDataStream()

        result_level = data_stream.level

        for stream_name in data_stream.streams():
            modeler = self.sub_components[stream_name]
            if modeler is None:
                result[stream_name] = data_stream[stream_name]
            else:
                result_datastream = modeler.reconstruct_datastream(
                    data_stream[stream_name], inference=inference)
                result[stream_name] = result_datastream
                result_level = result_datastream.level

        result.level = result_level

        return result
