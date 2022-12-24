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

        if self.hyper_params.machine.serialize and self.hyper_params.machine.serialize.load:
            machine_data = TMDataStream.deserialize(self.hyper_params.machine.serialize.path,
                                                   self.hyper_params.machine)
            logger.info(f"serialized machine data loaded from {self.hyper_params.machine.serialize} ")
            return machine_data

        if self.hyper_params.problem.serialize and self.hyper_params.problem.serialize.load:
            problem_data = TMDataStream.deserialize(self.hyper_params.problem.serialize.path,
                                                   self.hyper_params.problem)
            logger.info(f"serialized problem data loaded from {self.hyper_params.problem.serialize} ")
            return problem_data

        if self.hyper_params.task.serialize and self.hyper_params.task.serialize.load:
            task_data = TMDataStream.deserialize(self.hyper_params.task.serialize.path,
                                                self.hyper_params.task)
            logger.info(f"serialized task data loaded from {self.hyper_params.task.serialize}")
            return task_data


        if "task" in self.hyper_params:
            return self.DataStreamType(self.hyper_params.task)

        if "problem" in self.hyper_params:
            return self.DataStreamType(self.hyper_params.problem)

        if "machine" in self.hyper_params:
            return self.DataStreamType(self.hyper_params.machine)

        return self.DataStreamType(None)


class TMOfflineOutputStream(TMConfigurable):

    def __init__(self, hyper_params):

        super().__init__(hyper_params)

    def write(self, datastream):
        if self.hyper_params.serialize and self.hyper_params.serialize.save:
            datastream.serialize(self.hyper_params.serialize.path)

