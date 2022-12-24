from tripmaster.core.concepts.component import TMConfigurable, TMComponent
from tripmaster.core.concepts.data import TMDataChannel, TMDataLevel, TMDataStream
from tripmaster import logging

logger = logging.getLogger(__name__)

class TMPipelineUnit(TMComponent):

    Uri: str = None

    def __init__(self, hyper_params):
        super().__init__(hyper_params)


    def forward(self, input_data):

        pass


class TMSystemPipelineUnit(TMPipelineUnit):

    SystemType = None

    def __init__(self, hyper_params):
        super().__init__(hyper_params)

        assert not self.SystemType.is_learning()

        self.system = self.SystemType.create(self.hyper_params.system)

    def forward(self, input_data):

        data_channel = TMDataChannel(data=input_data, level=TMDataLevel.Task)
        data_stream = TMDataStream(level=TMDataLevel.Task)
        data_stream.inference_channels = ["inference"]
        data_stream["inference"] = data_channel

        inference_data_stream = self.system.run(data_stream, self.hyper_params.job)

        return list(inference_data_stream["inference"])

class TMPipelineUnitFactory(object):

    instance = None

    def __init__(self):

        self.uri_unit_map = dict()

    def register(self, unit):

        self.uri_unit_map[unit.Uri] = unit

    def create(self, uri, hyper_params):

        unit_type = self.uri_unit_map[uri]
        return unit_type(hyper_params)

    def units(self):

        for unit_uri in self.uri_unit_map.keys():
            yield unit_uri

    @classmethod
    def get(cls):
        """

        Returns:

        """
        if cls.instance is None:
            cls.instance = TMPipelineUnitFactory()

        return cls.instance


class TMPipeline(TMConfigurable):

    def __init__(self, hyper_params, pipelines):

        super().__init__(hyper_params)

        self.pipelines = []

        self.pipeline_factory = TMPipelineUnitFactory.get()

        supported_units = list(self.pipeline_factory.units())

        for pipeline_uri in pipelines:

            if not pipeline_uri in supported_units:
                message = f"Unsupported unit：{pipeline_uri}. Supported units: {supported_units}"
                logger.error(message)
                raise Exception(message)
            unit = self.pipeline_factory.create(pipeline_uri, self.hyper_params.units[pipeline_uri])
            self.pipelines.append(unit)

    def forward(self, input_data):

        for unit in self.pipelines:
            output_data = unit.forward(input_data)
            for input_item, output_item in zip(input_data, output_data):
                input_item.update(output_item)

        return input_data



