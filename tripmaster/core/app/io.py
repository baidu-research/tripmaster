from tripmaster.core.concepts.component import TMConfigurable
from tripmaster.core.concepts.data import TMDataStream

from tripmaster import logging

logger = logging.getLogger(__name__)

class TMApplicationIOStream(TMConfigurable):
    """
    TMApplicationIO
    """

    DataStreamType = None

    def data_stream(self):
        raise NotImplementedError()


class TMOfflineInputStream(TMApplicationIOStream):
    """
    TMOfflineApplicationInput
    """

    def data_stream(self):

        if "task" in self.hyper_params:
            return self.DataStreamType.create(self.hyper_params.task)

        if "problem" in self.hyper_params:
            return self.DataStreamType.create(self.hyper_params.problem)

        if "machine" in self.hyper_params:
            return self.DataStreamType.create(self.hyper_params.machine)

        return self.DataStreamType(None)


class TMOfflineOutputStream(TMConfigurable):

    def __init__(self, hyper_params):

        super().__init__(hyper_params)

    def write(self, datastream):
        if self.hyper_params.serialize and self.hyper_params.serialize.save:
            datastream.serialize(self.hyper_params.serialize.save)

