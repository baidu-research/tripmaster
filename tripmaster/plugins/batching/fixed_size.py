from tripmaster.core.components.operator.strategies.batching import TMBatchingStrategy


from tripmaster.core.components.backend import TMBackendFactory


B = TMBackendFactory.get().chosen()


class TMFixedBatchBatchingStrategy(TMBatchingStrategy):

    Name = "fixed_size"

    def batch_sampler(self, data_channel, data_traits, learning=True, distributed=False):


        if learning:
            initial_sampler = B.RandomSampler(data_channel)
        else:
            initial_sampler = B.SequentialSampler(data_channel)
        batch_sampler_ = B.BatchSampler(
            initial_sampler, batch_size=self.hyper_params.batch_size,
            drop_last=self.hyper_params.drop_last)

        if distributed:
            batch_sampler_ = B.DistributedBatchSampler(batch_sampler_, shuffle=False)

        return batch_sampler_

