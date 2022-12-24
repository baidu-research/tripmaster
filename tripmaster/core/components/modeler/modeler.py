"""
TM problem modeler
"""
import abc
import itertools
import numbers
from collections import OrderedDict, defaultdict
from enum import Enum, auto
from typing import Mapping, Type, Union

from tripmaster.core.concepts.component import TMComponent, TMSerializableComponent, \
    TMMultiComponentMixin
from tripmaster import logging
from tripmaster.core.concepts.contract import TMContractChannel
from tripmaster.core.concepts.data import TMDataStream, TMDataLevel, TMDataChannel, TMMultiDataStream
from tripmaster.core.concepts.modeler import TMModelerInterface, TMMultiModelerInterface
from tripmaster.core.concepts.scenario import TMScenario
from tripmaster.core.concepts.schema import TMSchema

logger = logging.getLogger(__name__)


def merge_reconstructs(result1, result2):
    """

    Args:
        result1 ():
        result2 ():

    Returns:

    """

    if type(result1) != type(result2):
        raise Exception(f"type not equal: {type(result1)} != {type(result2)}")
    if isinstance(result1, (str, numbers.Number)):
        if result1 != result2:
            raise Exception(f"numerical/literal value not equal : {result1} vs {result2}")
        else:
            return result1
    elif isinstance(result1, set):
        raise Exception("unable to merge set type values")
    elif isinstance(result1, list):
        if len(result1) != len(result2):
            raise Exception(f"list length not equal for : "
                            f"{len(result1)} != {len(result2)}")
        return [merge_reconstructs(x, y) for x, y in zip(result1, result2)]
    elif isinstance(result1, dict):

        for key in result2:
            if key not in result1:
                result1[key] = result2[key]
            else:
                result1[key] = merge_reconstructs(result1[key], result2[key])
        return result1


def add_sample_uri(stream, key):

    for idx, x in enumerate(stream):
        x[key] = idx
        yield x

class TMModeler(TMSerializableComponent, TMModelerInterface):
    """
    TMProblemModeler
    """

    def __init__(self, hyper_params, states=None):
        super().__init__(hyper_params, states=states)

        self.upstream_contract = None
        self.downstream_contract = None

    def set_contract(self, upstream_contract, downstream_contract):
        """

        Args:
            upstream_contract:
            downstream_contract:

        Returns:

        """

        self.upstream_contract = upstream_contract
        self.downstream_contract = downstream_contract

    def push_history(self, output, input, input_level):
        """

        Args:
            output:
            input:
            input_level:

        Returns:

        """

        forward_request = self.requires(forward=True).validate(input)
        forward_provide = self.provides(forward=True).validate(output)
        push_keys = set(forward_request.keys()) - set(forward_provide.keys())

        history_not_used_keys = set(key for key in input.keys()
                                    if key not in forward_request.keys() and key not in forward_provide and "@" not in key)

        for key in list(input.keys()):
            if key in push_keys or key in history_not_used_keys:
                output[f"{key}@{input_level}"] = input[key]  # move the inputs to upper layer
                if output is input:
                    del output[key]

        if output is not input:
            for key in list(input.keys()): # output may reuse the input dict
                is_history = any(key.endswith(f"@{x}") for x in TMDataLevel.upper_level(input_level))

                if is_history:  # historical layer
                    output[key] = input[key]

        return output

    def pop_history(self, source_samples, input_level):
        """

        Args:
            output:
            input:
            input_level:

        Returns:

        """

        target_level = TMDataLevel.reconstruct(input_level)
        uri_keys = set(TMDataLevel.uri_key(l) for l in TMDataLevel.upper_level(input_level))
        key_map = dict()
        remove_keys = set()
        for key in list(source_samples[0].keys()):

            # if key in uri_keys:
            #     key_map[key] = key
            #
            if key.endswith(f"@{target_level}") and key not in uri_keys:
                new_key = key.rsplit("@", 1)[0]
                key_map[key] = new_key

        # if inference:
        #     required_keys = set(self.requires(forward=False).validate(source_samples[0]).keys())
        # else:
        #     required_keys = set()

        target_samples = []
        for sample in source_samples:

            for key in list(sample.keys()):

                # if "@" not in key and key not in required_keys:
                #     del sample[key]

                if key in key_map:
                    sample[key_map[key]] = sample[key]
                    del sample[key]

                # if key in uri_keys:
                #     sample[key] = sample[key]
#            target_samples.append(target_sample)

        return source_samples

    def clean_reconstructed(self, results, input_level, inference):

        if inference:
            reconstructed = set(self.provides(forward=False).validate(results).keys())
        else:
            reconstructed = set()

        lower_keys = set(TMDataLevel.uri_key(l) for l in TMDataLevel.lower_level(input_level))

        for key in list(results.keys()):

            if key in lower_keys or key == TMDataLevel.uri_key(input_level):
                del results[key]

            if key not in reconstructed and "@" not in key:
                del results[key]

        return results

    def model_sample(self, sample, level: TMDataLevel, scenario: TMScenario):

        if self.upstream_contract:
            contracted_sample = self.upstream_contract.forward(sample)
        else:
            contracted_sample = sample

        if self.validate:
            for key in (TMContractChannel.Source, TMContractChannel.Target):
                assert self.requires(forward=True, channel=key).is_valid(contracted_sample)

        for result in self.model(contracted_sample, scenario):
            if self.validate:
                for key in (TMContractChannel.Source, TMContractChannel.Target):
                    assert self.provides(forward=True, channel=key).is_valid(result)

            result = self.push_history(result, sample, input_level=level)

            if self.downstream_contract:
                result = self.downstream_contract.forward(result)

            yield result

    def reconstruct_sample(self, samples, level: TMDataLevel,
                           scenario: TMScenario, with_truth=False):


        if self.downstream_contract:
            modeler_inner_samples = [self.downstream_contract.backward(x) for x in samples]
        else:
            modeler_inner_samples = list(samples)

        if self.validate and scenario == TMScenario.Inference:
            for x in modeler_inner_samples:
                for key in (TMContractChannel.Inference,):
                    assert self.requires(forward=False, channel=key).is_valid(x)

        target_samples = self.pop_history(modeler_inner_samples, input_level=level)
        result = self.reconstruct(target_samples, scenario=scenario, with_truth=with_truth)

        if self.validate and scenario == TMScenario.Inference:
            for key in (TMContractChannel.Inference,):
                assert self.provides(forward=False, channel=key).is_valid(result)

        if self.upstream_contract:
            result = self.upstream_contract.backward(result)

        return result

    def model_datachannel(self, data_channel: TMDataChannel, scenario: TMScenario):
        """
        Args:
            samples ():
            display_view ():
            sampler ():
            batch_size ():
        Returns:
        """

        processed = []

        for idx, sample in enumerate(data_channel):

            for result in self.model_sample(sample, scenario=scenario, level=data_channel.level):

                uri_key = TMDataLevel.uri_key(data_channel.level)
                result[uri_key] = idx
                processed.append(result)

        # make sure the problem samples for a raw sample are ordered together
        # processed.sort(key=lambda x: x[TMTaskDataStream.SAMPLE_URI_KEY])
        # processed = add_sample_uri(processed, TMProblemDataStream.SAMPLE_URI_KEY)
        return processed

    def reconstruct_datachannel(self, data_channel: TMDataChannel, scenario: TMScenario,
                                with_truth=False):
        """

        Args:
            problem_data:
            scenario:

        Returns:

        """

        data_iter = iter(data_channel)
        first_element = next(data_iter)
        uri_key = None
        for x in TMDataLevel.upper_level(data_channel.level):
            key = TMDataLevel.uri_key(x)
            if key in first_element:
                uri_key = key
                break

        data_iter = itertools.chain([first_element], data_iter)

        target_level = TMDataLevel.reconstruct(data_channel.level)
        uri_key = TMDataLevel.uri_key(target_level)

        def key_func(x):
            return x[uri_key]

        for key, group in itertools.groupby(data_iter,
                                            key=key_func):

            yield self.reconstruct_sample(group, level=data_channel.level,
                                          scenario=scenario, with_truth=with_truth)


    def model_datastream(self, datastream: TMDataStream, scenario: TMScenario):

        """

        Args:
            raw ():

        Returns:

        """
        channel_scenario_map = dict()
        if scenario == TMScenario.Learning:
            channel_scenario_map.update((x, TMScenario.Evaluation) for x in datastream.eval_channels)
            # if a channel occurs both in Evaluation and Learn, make sure it's scenario is set to Learn
            channel_scenario_map.update((x, TMScenario.Learning) for x in datastream.learn_channels)
        elif scenario == TMScenario.Evaluation:
            channel_scenario_map.update((x, TMScenario.Evaluation) for x in datastream.eval_channels)
        elif scenario == TMScenario.Inference:
            channel_scenario_map.update((x, TMScenario.Inference) for x in datastream.inference_channels)

        else:
            raise Exception(f"Unknown scenario {scenario}")

        target_stream = TMDataStream()
        target_stream.level = TMDataLevel.model(datastream.level)

        target_stream.learn_channels = datastream.learn_channels
        target_stream.eval_channels = datastream.eval_channels
        target_stream.inference_channels = datastream.inference_channels

        for channel, channel_scenario in channel_scenario_map.items():

            data_channel = datastream[channel]

            logger.info(f"Building {target_stream.level} dataset for channel {channel} in scenario {channel_scenario}")

            target_data_channel = self.model_datachannel(data_channel, scenario=scenario)

            target_stream[channel] = target_data_channel

            logger.info(f"Channel {channel} modeled")

        return target_stream

    def reconstruct_datastream(self, datastream: TMDataStream, scenario: TMScenario, with_truth=False):

        """
        two possible situation: 
        * reconstruction the truth stream, set inference = False
        * reconstruction the inferenced stream, set inference = True 
        Args:
            raw ():

        Returns:

        """

        target_stream = TMDataStream()

        target_stream.level = TMDataLevel.reconstruct(datastream.level)

        target_stream.learn_channels = datastream.learn_channels
        target_stream.eval_channels = datastream.eval_channels
        target_stream.inference_channels = datastream.inference_channels

        if scenario in {TMScenario.Learning, TMScenario.Evaluation}:
            target_channels = datastream.eval_channels
        elif scenario == TMScenario.Inference:
            target_channels = datastream.inference_channels
        else:
            raise Exception(f"Unknown scenario {scenario}")

        for channel in target_channels:

            target_datachannel = self.reconstruct_datachannel(datastream[channel],
                                                              scenario=scenario, with_truth=with_truth)
            target_stream[channel] = target_datachannel

        return target_stream

    def model_environment(self, env):
        """
        build a modeled environment from a raw environment
        Generally, environments from gym are encoded/modeled environments (with render to generate the raw images)
        But some environments (like text-world) are raw environments, so we need to model them
         for machine to learn
        """

        return env.apply_modeler(self)

    def reconstruct_environment(self, env):
        """
        reconstruct the raw environment from the modeled environment
        Honestly, I don't know why we need this function.
        I add it for the sake of symmetry and completeness
        """
        raise NotImplementedError()



class TMContractOnlyModeler(TMModeler):

    ForwardRequestSchema = {TMContractChannel.Source: {}, TMContractChannel.Target: {}}
    ForwardProvisionSchema = {TMContractChannel.Source: {}, TMContractChannel.Target: {}}

    BackwardRequestSchema = {TMContractChannel.Inference: {}}
    BackwardProvisionSchema = {TMContractChannel.Inference: {}}

    def model(self, data, channel="learn", with_truth=False):

        yield self.upstream_contract.forward(data) if self.upstream_contract else data

    def reconstruct(self, samples, with_truth=True, inference=False):

        assert len(samples) == 1

        return self.upstream_contract.backward(samples[0]) if self.upstream_contract else samples[0]

TMContractOnlyModeler.init_class()


import itertools


class TMSharedModeler(TMMultiModelerInterface, TMModeler):
    """
    MultiTask Problem Modeler
    """


    def __init__(self, hyper_params, states=None, default_init=True):

        TMModeler.__init__(self, hyper_params, states=states)
        TMMultiComponentMixin.__init__(self, hyper_params, states=states, default_init=default_init)

        # if default_init:
        #     for task in self.sub_components:
        #         if self[task] is None:
        #             self[task] = TMContractOnlyModeler(None)


    def model(self, data, channel="learn", with_truth=False):
        """

        Args:
            inputs ():

        Returns: the modeled data together with the inputs.
            The inputs is included because the reconstruct may need them

        """
        # results = []
        # for result_items in zip(modeler.model(data, channel=channel, with_truth=with_truth)
        #     for task, modeler in self.modelers.items()):
        #     result = dict(itertools.chain(*[x.items() for x in result_items]))
        #     results.append(result)

        task_results = {}

        for task, modeler in self.sub_components.items():
            if modeler is None:
                task_results[task] = [data[task]]
            else:
                result = modeler.model(data[task], channel=channel, with_truth=with_truth)
                task_results[task] = list(result)

        tasks = list(self.sub_components.keys())
        for data in zip(*(task_results[x] for x in tasks)):
            sample = dict(zip(tasks, data))
            yield sample

    def reconstruct(self, samples, with_truth=True, inference=False):
        """

        Args:
            model ():

        Returns:

        """
        tasks = list(self.sub_components.keys())
        task_samples = defaultdict(list)

        for sample in samples:
            for key in tasks:
                task_samples[key].append(sample[key])

        sample = dict()
        for task, modeler in self.sub_components.items():
            if modeler is None:
                assert len(task_samples) == 1
                sample[task] = task_samples[task][0]
            else:
                result = modeler.reconstruct(task_samples[task], with_truth=with_truth, inference=inference)
                sample[task] = result

        return sample



class TMMultiModeler(TMMultiModelerInterface, TMModeler):
    """
    MultiTask Problem Modeler
    """

    def __init__(self, hyper_params, states=None, default_init=True):
        TMModeler.__init__(self, hyper_params, states=states)
        TMMultiComponentMixin.__init__(self, hyper_params, states=states, default_init=default_init)

        # if default_init:
        #     for task in self.sub_components:
        #         if self[task] is None:
        #             self[task] = TMContractOnlyModeler(None)

    def model(self, data, channel="learn", with_truth=False):
        pass

    def reconstruct(self, samples, with_truth=True, inference=False):
        pass

    def model_datastream(self, data_stream: TMMultiDataStream, scenario):
        """

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
                result_datastream = modeler.model_datastream(data_stream[stream_name], scenario=scenario)
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
                result_datastream = modeler.reconstruct_datastream(data_stream[stream_name], inference=inference)
                result[stream_name] = result_datastream
                result_level = result_datastream.level

        result.level = result_level

        return result