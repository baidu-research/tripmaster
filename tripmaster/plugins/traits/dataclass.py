import dataclasses

from tripmaster.core.components.machine.data_traits import TMElementTraits, TMElementTraitsFactory, UnsupportedTraitsError, \
    TMElementBatchTraits
import numpy as np

from tripmaster.core.components.backend import TMBackendFactory

B = TMBackendFactory.get().chosen()


class TMDataClassElementTraits(TMElementTraits):

    ElementType = dataclasses.is_dataclass

    @classmethod
    def collate(self, list_of_samples):
        first_elem = list_of_samples[0]
        field_dict = dict()
        for field in dataclasses.fields(first_elem):
            this_value_list = [getattr(x, field.name) for x in list_of_samples]

            try:
                traits = TMElementTraitsFactory.get().get_element_traits(this_value_list)
                field_dict[field.name] = traits.collate(this_value_list)
            except UnsupportedTraitsError as e:
                field_dict[field.name] = this_value_list

        return first_elem.__class__(**field_dict)


class TMDataClassElementBatchTraits(TMElementBatchTraits):

    ElementBatchType = dataclasses.is_dataclass

    @classmethod
    def decollate(cls, batch):

        batch_size = cls.batch_size(batch)
        field_dict = dict()
        for field in dataclasses.fields(batch):
            value = getattr(batch, field.name)
            try:
                traits = TMElementTraitsFactory.get().get_element_batch_traits(value)
                field_dict[field.name] = traits.decollate(value)
            except UnsupportedTraitsError as e:
                field_dict[field.name] = value

        return [batch.__class__(** dict((k, field_dict[k][i]) for k in field_dict)) for i in range(batch_size)]

    @classmethod
    def batch_size(cls, batch):

        for field in dataclasses.fields(batch):
            value = getattr(batch, field.name)
            try:
                traits = TMElementTraitsFactory.get().get_element_batch_traits(value)
                return traits.batch_size(value)
            except UnsupportedTraitsError as e:
                raise

        raise Exception(f"Cannot calculate the batch size")

    @classmethod
    def to_device(cls, batch, device):

        field_dict = dict()
        for field in dataclasses.fields(batch):
            this_value = getattr(batch, field.name)

            try:
                traits = TMElementTraitsFactory.get().get_element_batch_traits(this_value)
                field_dict[field.name] = traits.to_device(this_value, device)
            except UnsupportedTraitsError as e:
                field_dict[field.name] = this_value

        return batch.__class__(**field_dict)

    @classmethod
    def shape(cls, batch):
        shape_dict = dict()

        for field in dataclasses.fields(batch):
            this_value = getattr(batch, field.name)

            try:
                traits = TMElementTraitsFactory.get().get_element_batch_traits(this_value)
                shape_dict[field.name] = traits.shape(this_value)
            except UnsupportedTraitsError as e:
                pass

        return shape_dict