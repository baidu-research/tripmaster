
from tripmaster.core.components.machine.data_traits import TMElementTraits, TMElementTraitsFactory, UnsupportedTraitsError, \
    TMElementBatchTraits
import numpy as np

from tripmaster.core.components.backend import TMBackendFactory
from tripmaster.plugins.traits.basic import TMTensorElementTraits

B = TMBackendFactory.get().chosen()

import dgl


class TMDGLElementTraits(TMElementTraits):

    ElementType = dgl.DGLGraph

    @classmethod
    def collate(self, list_of_samples):
        return dgl.batch(list_of_samples)


class TMDGLElementBatchTraits(TMElementBatchTraits):

    ElementBatchType = dgl.DGLGraph

    @classmethod
    def decollate(self, batch):
        return dgl.unbatch(batch)

    @classmethod
    def batch_size(cls, batch):

        return len(batch.batch_num_nodes())

    @classmethod
    def to_device(cls, batch, device):

        return batch.to(device)

    @classmethod
    def shape(cls, batch):

        return {"nodes": batch.batch_num_nodes(), "edges": batch.batch_num_edges()}