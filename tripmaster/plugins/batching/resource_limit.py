import sys
import math
import numpy as np

from tripmaster.core.components.operator.strategies.batching import TMBatchingStrategy


from tripmaster.core.components.backend import TMBackendFactory

B = TMBackendFactory.get().chosen()


def identity(x):
    """

    Args:
        x ():

    Returns:

    """
    return x


class SortedSampler(B.Sampler):
    """ Samples elements sequentially, always in the same order.

    Args:
        data (iterable): Iterable data.
        sort_key (callable): Specifies a function of one argument that is used to extract a
            numerical comparison key from each list element.

    Example:
        >>> list(SortedSampler(range(10), sort_key=lambda i: -i))
        [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]

    """

    def __init__(self, data, sort_key=identity):
        super().__init__(data)
        self.data = data
        self.sort_key = sort_key
        zip_ = [(i, self.sort_key(row)) for i, row in enumerate(self.data)]
        zip_ = sorted(zip_, key=lambda r: r[1])
        self.sorted_indexes = [item[0] for item in zip_]

    def __iter__(self):
        return iter(self.sorted_indexes)

    def __len__(self):
        return len(self.data)


class BucketBatchSampler(B.BatchSampler):
    """ `BucketBatchSampler` toggles between `sampler` batches and sorted batches.

    Typically, the `sampler` will be a `RandomSampler` allowing the user to toggle between
    random batches and sorted batches. A larger `bucket_size_multiplier` is more sorted and vice
    versa.

    Background:
        ``BucketBatchSampler`` is similar to a ``BucketIterator`` found in popular libraries like
        ``AllenNLP`` and ``torchtext``. A ``BucketIterator`` pools together examples with a similar
        size length to reduce the padding required for each batch while maintaining some noise
        through bucketing.

        **AllenNLP Implementation:**
        https://github.com/allenai/allennlp/blob/master/allennlp/data/iterators/bucket_iterator.py

        **torchtext Implementation:**
        https://github.com/pytorch/text/blob/master/torchtext/data/iterator.py#L225

    Args:
        sampler (torch.data.utils.sampler.Sampler):
        batch_size (int): Size of mini-batch.
        drop_last (bool): If `True` the sampler will drop the last batch if its size would be less
            than `batch_size`.
        sort_key (callable, optional): Callable to specify a comparison key for sorting.
        bucket_size_multiplier (int, optional): Buckets are of size
            `batch_size * bucket_size_multiplier`.

    Example:

        >>>
        >>> from torch.utils.data.sampler import SequentialSampler
        >>> sampler = SequentialSampler(list(range(10)))
        >>> list(BucketBatchSampler(sampler, batch_size=3, drop_last=False))
        [[6, 7, 8], [0, 1, 2], [3, 4, 5], [9]]
        >>> list(BucketBatchSampler(sampler, batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self,
                 sampler,
                 batch_size,
                 drop_last,
                 sort_key=identity,
                 bucket_size_multiplier=100):
        super().__init__(sampler=sampler, batch_size=batch_size, drop_last=drop_last)
        self.sort_key = sort_key
        self.bucket_sampler = B.BatchSampler(sampler,
                                           min(batch_size * bucket_size_multiplier, len(sampler)),
                                           False)

    def __iter__(self):
        for bucket in self.bucket_sampler:
            zip_ = [(i, self.sort_key(i)) for i in bucket]
            zip_ = sorted(zip_, key=lambda r: r[1])
            sorted_indexes = [item[0] for item in zip_]

            #            sorted_sampler = SortedSampler(bucket, self.sort_key)
            #            # here sorted_sampler contains the sorted index of bucket
            for batch in B.SubsetRandomSampler(
                    list(B.BatchSampler(sampler=sorted_indexes, batch_size=self.batch_size, drop_last=self.drop_last))):
                yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return math.ceil(len(self.sampler) / self.batch_size)


def requested_resources(requests, padded_length=None):
    """

    Args:
        request:

    Returns:

    """
    new_requests = []

    if padded_length:
        idx = 0
        for request, padding_dim in requests:
            if padding_dim:
                request = list(request)
                request[padding_dim] = padded_length[idx]
                request = tuple(request)
                new_requests.append((request, padding_dim))
                idx += 1
            else:
                new_requests.append((request, padding_dim))
    else:
        new_requests = requests

    import math
    amount_of_request = 0
    for request, padding_dim in new_requests:
        amount_of_request += math.prod(request)

    return amount_of_request


def resource_keys(resource_requests):
    """

    Args:
        resource_request:

    Returns:

    """
    keys = []
    for request, padding_dim in resource_requests:
        if padding_dim:
            keys.append(request[padding_dim])

    return tuple(keys)


class TMMemoryManager(object):

    def __init__(self, data_traits, channel):

        padding_dim = data_traits.padding_dim()

        memory_request = [data_traits.memory_request(x) for x in channel.data]

        keys = list(padding_dim.keys())
        variable = []
        fixed = []

        for key in keys:
            request = np.array([x[key] for x in memory_request])
            if padding_dim[key] is not None:
                dim = padding_dim[key]
                variable_size = request[:, dim]
                fixed_size = request.prod(axis=1) / variable_size

            else:
                fixed_size = request.prod(axis=1)
                variable_size = np.ones_like(fixed_size)

            variable.append(variable_size)
            fixed.append(fixed_size)

        varaible_requests = np.stack(variable)
        fixed_requests = np.stack(fixed)

        self.keys = keys

        self.fixed_requests = fixed_requests
        self.variable_requests = varaible_requests

    def memory(self, index):
        return np.dot(self.variable_requests[:, index], self.fixed_requests[:, index])

    def padded_variable_size(self, batch_ids):
        return self.variable_requests[:, batch_ids].max(axis=1)

    def batch_fixed_size(self, batch_ids):
        return self.fixed_requests[:, batch_ids].sum(axis=1)

    def batch_memory_request(self, batch_ids):
        padded_variable_size = self.padded_variable_size(batch_ids)
        fixed_size = self.batch_fixed_size(batch_ids)

        return np.dot(padded_variable_size, fixed_size)


class ResourceLimitBucketBatchSampler(B.Sampler):
    """
    DynamicBucketBatchSampler
    """

    def __init__(self,
                 sampler,
                 memory_manager,
                 resource_limit,
                 resource_allocation_range=10000,
                 drop_last=False,
                 sort_key=identity,
                 random=True
                 ):
        self.memory_manager = memory_manager
        self.sampler = sampler
        self.resource_limit = resource_limit
        self.resource_allocation_range = resource_allocation_range
        self.drop_last = drop_last
        self.sort_key = sort_key
        self.random = random

        self.bucket_sampler = B.BatchSampler(sampler,
                                           self.resource_allocation_range,
                                           False)

    def __iter__(self):
        for bucket in self.bucket_sampler:
            zip_ = [(i, self.sort_key(i)) for i in bucket]
            zip_ = sorted(zip_, key=lambda r: r[1])
            sorted_indexes = [item[0] for item in zip_]

            resource_batches = [x for x in ResourceLimitBatchSampler(sampler=sorted_indexes,
                                                                     memory_manager=self.memory_manager,
                                                                     resource_limit=self.resource_limit,
                                                                     drop_last=self.drop_last)]
            if self.random:
                for batch in B.SubsetRandomSampler(resource_batches):
                    yield batch
            else:
                for batch in resource_batches:
                    yield batch

    def __len__(self):
        raise NotImplementedError()


class ResourceLimitBatchSampler(B.Sampler):
    r"""
    ResourceLimitBatchSampler
    """

    def __init__(self, sampler: B.Sampler,
                 memory_manager,
                 resource_limit: int, drop_last: bool) -> None:
        super().__init__(None)

        self.sampler = sampler
        self.memory_manager = memory_manager
        self.resource_limit = resource_limit
        self.drop_last = drop_last

        #        if any(requested_resources(x) > . for x in resource_requests):
        #            raise Exception("Some resource request already exceed the resource limit.")

        self.generate_batches()

    def generate_batches(self):

        self.batch_ids = []
        batch = []
        max_padded_resource_len = None
        for idx, sample_id in enumerate(self.sampler):
            if not batch:
                batch = [sample_id]
                total_requested_resource = self.memory_manager.batch_memory_request(batch)
                if total_requested_resource > self.resource_limit:
                    raise Exception(
                        f"one sample {sample_id} resource request exceed the resource limits {self.resource_limit} ")
                continue

            new_batch = batch + [sample_id]

            total_requested_resource = self.memory_manager.batch_memory_request(new_batch)

            if total_requested_resource <= self.resource_limit:
                batch = new_batch
            else:
                self.batch_ids.append(batch)

                batch = [sample_id]

        if len(batch) > 0 and not self.drop_last:
            self.batch_ids.append(batch)

    def __iter__(self):

        for batch in self.batch_ids:
            yield batch

        self.finish_iteration = True

    def __len__(self):

        return len(self.batch_ids)

class TMResourceLimitBatchingStrategy(TMBatchingStrategy):

    Name = "resource_limit"


    def batch_sampler(self, data_channel, data_traits, learning=True, distributed=False):

        memory_manager = TMMemoryManager(data_traits, data_channel)

        if learning:
            initial_sampler = B.RandomSampler(data_channel)
            sort_key = memory_manager.memory
            resource_limit = self.hyper_params.learning_memory_limit
            resource_allocation_range = self.hyper_params.resource_allocation_range
        else:
            initial_sampler = B.SequentialSampler(data_channel)
            sort_key = identity
            resource_limit = self.hyper_params.inferencing_memory_limit
            resource_allocation_range = sys.maxsize

        batch_sampler_ = ResourceLimitBucketBatchSampler(
            initial_sampler,
            memory_manager=memory_manager,
            resource_limit=resource_limit,
            resource_allocation_range=resource_allocation_range,
            sort_key=sort_key,
            drop_last=self.hyper_params.drop_last,
            random=learning
        )
        if distributed:
            batch_sampler_ = B.DistributedBatchSampler(batch_sampler_, shuffle=False)

        return batch_sampler_

