"""
TM supervised operator
"""
import weakref
from typing import Union
from more_itertools import ichunked

from tqdm import tqdm

from tripmaster.core.components.evaluator import MachineEvaluationStreamInfo
from tripmaster.core.components.operator.operator import deep_merge_dict, TMEvaluatorMixin, TMOperator, \
    TMLearnerMixin, TMEvaluatorMixin
from tripmaster.core.components.operator.strategies.model_selection import BestOneModelSelectionStrategy
from tripmaster.core.components.operator.strategies.optimization import EpochwiseLRUpdateStrategy
from tripmaster.core.concepts.contract import TMContractChannel
from tripmaster import logging
from tripmaster.core.concepts.data import TMDataStream, TMDataLevel, TMMultiDataStream
from tripmaster.core.concepts.hyper_params import TMHyperParams
from tripmaster.core.concepts.scenario import TMScenario
from tripmaster.utils.stream import isolate_iterators
from tripmaster import P, M, T, D
logger = logging.getLogger(__name__)

class TMSuperviseOperator(TMOperator):
    """
    TMSupervisedOperator
    """


    def fit_memory(self, machine_samplestream: Union[TMDataStream, TMMultiDataStream]):

        assert machine_samplestream.level == TMDataLevel.Machine

        memory_samplestream = self.memory_modeler.model_datastream(machine_samplestream,
                                                                   scenario=self.scenario)

        return memory_samplestream

    def unfit_memory(self, memory_samplestream: Union[TMDataStream, TMMultiDataStream],
                     scenario: TMScenario, with_truth=False):

        machine_samplestream = self.memory_modeler.reconstruct_datastream(memory_samplestream,
                                                                          scenario=scenario)

        return machine_samplestream


    def batchify(self, machine_stream: TMDataStream):
        """

        Args:
            problem_dataset:

        Returns:

        """
        machine_batchstream = self.batch_modeler.model_datastream(machine_stream,
                                                                  scenario=self.scenario)

        return machine_batchstream

    def unbatchify(self, machine_batchstream: TMDataStream, scenario:TMScenario, with_truth=False):
        """

        Args:
            problem_dataset:

        Returns:

        """
        machine_samplestream = self.batch_modeler.reconstruct_datastream(machine_batchstream,
                                                                         scenario=scenario,
                                                                         with_truth=with_truth)

        return machine_samplestream




class TMSupervisedEvaluatorMixin(TMEvaluatorMixin):
    """
    TMSupervisedEvaluatorMixin
    """
    def __init__(self, hyper_params,  **kwargs):
        super().__init__(hyper_params, **kwargs)



    def evaluate_channel(self, data_loader, channel, local_rank, epoch):
        """evaluate(

        Args:
            local_rank:
            data_loader:

        Returns:

        """

        P.OptimizerBehaviors.set_inference_mode(self.machine)
        batch_traits = self.machine.BatchTraits

        with P.OptimizerBehaviors.no_grad():
            for i, truth in tqdm(enumerate(data_loader), desc=f"Evaluating Channel {channel}: "):
                # measure data loading time

                try:
                    truth = self.reallocate_data(truth, local_rank)

                    self.machine.validate_forward_input(truth, scenario=TMScenario.Evaluation)
                    inferenced = self.machine.forward(truth, scenario=TMScenario.Evaluation)
                    self.machine.validate_forward_output(inferenced, scenario=TMScenario.Evaluation)

                    # truth.update(inferenced)  # sometimes the truth is constructed by machine
                    # for key in truth:
                    #     if key.endswith("_id") or key.endswith("_uri") or key == "uri":
                    #         inferenced[key] = truth[key]
                    #     if key not in inferenced:
                    #         inferenced[key] = truth[key]
                    deep_merge_dict(inferenced, truth)

                    loss = self.machine.loss(inferenced, truth).detach()
                    batch_size = batch_traits.batch_size(truth)

                    yield {"objective": loss, "sample_num": batch_size}, truth, inferenced

                #                    info = EvaluationMachineBatchInferencedInfo(
                #                        truth_machine_batch=truth, inference_machine_batch=inferenced,
                #                        batch_loss=loss, channel=channel, local_rank=local_rank, device=device, epoch=self.epoch)

                #                    batch_inferenced_signal.send(info)

                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        shapes = batch_traits.data_shape(truth)
                        logger.error(f"Out of Memory: Data Shapes = {shapes}")
                    # logger.exception(e)
                    raise e
                except Exception as e:
                    # logger.exception(e)
                    raise e

        # avg_loss = total_loss / sample_num if sample_num > 0 else 0
        #
        # loss_info = TMEvaluationSignals.MachineLossInfo(channel=channel, local_rank=local_rank,
        #                                                    loss=avg_loss, epoch=epoch)
        # signal(TMEvaluationSignals.ON_MACHINE_LOSS_COMPUTED).send(loss_info)

    def evaluate(self, batchstreams: TMDataStream, local_rank, epoch, step):

        # if local_rank != 0:
        #     return

        if not batchstreams.eval_channels:
            return

        truth_machine_datastream = TMDataStream()
        truth_machine_datastream.level = TMDataLevel.Batch

        truth_machine_datastream.eval_channels = batchstreams.eval_channels

        inference_machine_datastream = TMDataStream()
        inference_machine_datastream.level = TMDataLevel.Batch
        inference_machine_datastream.eval_channels = batchstreams.eval_channels
        channeled_loss_streams = dict()

        for channel in batchstreams.eval_channels:
            loss_stream, truth_stream, inferenced_stream = isolate_iterators(
                self.evaluate_channel(batchstreams[channel], channel, local_rank, epoch),
                3
            )
            channeled_loss_streams[channel] = loss_stream
            truth_machine_datastream[channel] = truth_stream
            inference_machine_datastream[channel] = inferenced_stream

        device = self.device(local_rank)

        info = MachineEvaluationStreamInfo(objective_stream=channeled_loss_streams,
                                           truth_stream=truth_machine_datastream,
                                           inferenced_stream=inference_machine_datastream,
                                           local_rank=local_rank, device=device, epoch=epoch, step=step)
        evaluation_results = self.evaluate_signal.send(info)

        return evaluation_results


def train_worker(local_rank, learner, data_streams, runtime_options):
    """

    Args:
        local_rank ():
        learner ():
        data_streams ():
        evaluator ():

    Returns:

    """
    logger.warning(f"start trainer {local_rank}")

    learner.train(data_streams, runtime_options, local_rank)


class TMSupervisedLearnerMixin(TMLearnerMixin):
    """
    TMSupervisedLearnerMixin add Learning ability to TMSupervisedOperator
    """


    def train_step(self, local_rank, channel, train_batchstreams):
        """

        Args:
            local_rank:
            data_loader:

        Returns:

        """

        # switch to train mode
        P.OptimizerBehaviors.set_train_mode(self.machine)

        batch_traits = self.machine.BatchTraits

        total_loss = 0.0
        total_sample_num = 0
        with tqdm(desc=f"Channel {channel}, Epoch {self.epoch}", leave=False,
                  postfix=dict(batch_size=0, batch_loss=0, average_loss=0), unit="batch") as t:
            
            for i, batch in enumerate(train_batchstreams[channel]):
                # measure data loading time
                self.optimization_strategy.on_batch_start(self.machine, batch, i)

                if self.optimization_strategy.finish():
                    logger.info("optimizer terminate criterion satisfied. Optimization Finish.")
                    break

                try:
                    batch = self.reallocate_data(batch, local_rank)
                    # print( batch.place )
                    # assert(1==2)
                    batch_size = batch_traits.batch_size(batch)
                    # print("batch_size: ", batch_size)
                    
                    
                    #               logger.warning(f"batch size = {batch_size}") #, hyper_param == {self.hyper_params}, world_size = {self.world_size}")

                    # if training_setting.parallel == "dp" and world_size * 2 > batch_size:
                    #     duplicate_num = (world_size * 2 + (batch_size - 1)) // batch_size
                    #     logger.warning(
                    #         f"we are repeating {duplicate_num} times of small batch with size {batch_size} for dp ")
                    #     batch_traits.duplicate(batch, duplicate_num)
                    self.machine.validate_forward_input(batch, scenario=TMScenario.Learning)
                    output = self.machine.forward(batch, scenario=TMScenario.Learning)
                    self.machine.validate_forward_input(output, scenario=TMScenario.Learning)
                    # batch.update(output)  # sometimes the truth is generated by machine
                    # deep_merge_dict(batch, output)
                    loss = self.machine.loss(output, batch)

                    reduced_loss = self.distributed_strategy.sync_loss(loss)

                    self.optimization_strategy.on_batch_end(self.machine, output, reduced_loss, i)

                    total_loss += reduced_loss.detach().item() * batch_size
                    total_sample_num += batch_size
                    avg_loss = total_loss / total_sample_num if total_sample_num > 0 else 0.0
                    t.set_postfix(batch_loss=loss.item(), batch_size=batch_size, average_loss=avg_loss, refresh=False)
                    t.update()

                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        shapes = batch_traits.shape(batch)
                        batch_size = batch_traits.batch_size(batch)
                        logger.error(
                            f"Out of Memory for {i}-th Batch, Batch Size = {batch_size}, Data Shapes = {shapes}")

                    # logger.exception(e)
                    raise e

                except Exception as e:
                    shapes = batch_traits.shape(batch)
                    logger.error(f"Learn for {i}-th Batch, Data Shapes = {shapes}")
                    # logger.exception(e)
                    raise e

                self.step += 1
                if self.evaluation_trigger.trigger(self):
                    P.OptimizerBehaviors.set_inference_mode(self.machine)
                    self.eval_and_select_model(train_batchstreams, local_rank, self.epoch, self.step)
                    P.OptimizerBehaviors.set_train_mode(self.machine)

        return total_loss / total_sample_num if total_sample_num > 0 else 0.0

    def train(self, train_batchstreams: TMDataStream, runtime_options, local_rank):
        """

        Args:
            local_rank ():
            model ():
            optimization ():
            train_data_loader ():
            dev_data_loader ():
            test_data_loader ():
            problem_evaluator ():
            task_evaluator ():
            learn_conf ():

        Returns:

        """

        self.local_rank = local_rank

        # training_setting = self.optimization_strategy.hyper_params

        # self.runtime_options = runtime_options
        # if training_setting.seed:
        #     seed = training_setting.seed
        #     random.seed(seed)
        #     #            torch.manual_seed(seed)
        #     #            cudnn.deterministic = True
        #     warnings.warn('You have chosen to seed training. '
        #                   'This will turn on the CUDNN deterministic setting, '
        #                   'which can slow down your training considerably! '
        #                   'You may see unexpected behavior when restarting '
        #                   'from checkpoints.')

        self.distributed_strategy.init(local_rank)

        if runtime_options.mode == "eval":
            P.OptimizerBehaviors.set_inference_mode(self.machine)
            signal_returns = self.evaluate(train_batchstreams, local_rank, -1, -1)
            assert len(signal_returns) == 1
            evaluation_results = signal_returns[0][1]

            self.metric_logging_strategy.log(evaluation_results)

            P.OptimizerBehaviors.set_train_mode(self.machine)
            return

        # self.optimization_strategy.act_on(self.machine)
        self.epoch = 0
        self.step = 0
        while True:

            self.optimization_strategy.on_epoch_start(self.machine, self.epoch)

            if self.optimization_strategy.finish():
                logger.info("optimizer terminate criterion satisfied. Optimization Finish.")
                break

            for channel in train_batchstreams.learn_channels:
                loss = self.train_step(local_rank, channel, train_batchstreams)

            self.optimization_strategy.on_epoch_end(self.machine, self.epoch)

            self.epoch += 1
            if self.evaluation_trigger.trigger(self):
                P.OptimizerBehaviors.set_inference_mode(self.machine)
                self.eval_and_select_model(train_batchstreams, local_rank, self.epoch, self.step)
                P.OptimizerBehaviors.set_train_mode(self.machine)

class TMSuperviseLearner(TMSuperviseOperator,
                            TMSupervisedEvaluatorMixin,
                            TMSupervisedLearnerMixin):
    """
    Supervised Learning Operator
    """

    def __init__(self, hyper_params, machine, states=None, **kwargs):
        super().__init__(hyper_params=hyper_params, machine=machine, states=states,
                         scenario=TMScenario.Learning, **kwargs)

    def operate(self, batch_stream: TMDataStream, runtime_options):
        """

        Args:
            problem ():
            raw ():
            problem_modeler ():

        Returns:

        """

        # logger.info(f"operate with parameter {self.runtime_options}")
        assert batch_stream.level == TMDataLevel.Batch, f"wrong data level, {batch_stream.level}"

        # memory_stream = self.fit_memory(machine_stream)
        # batch_stream = self.batchify(memory_stream)

        self.distributed_strategy.run(train_worker, batch_stream, runtime_options)

    def test(self, config):
        TMSupervisedLearnerMixin.test(self, config)



def predict_worker(local_rank, predictor, data_streams, runtime_options):
    """

    Args:
        local_rank ():
        learner ():
        data_streams ():
        evaluator ():

    Returns:

    """
    logger.warning(f"start inference {local_rank}")

    result = predictor.inference(local_rank, data_streams, runtime_options)
    return result


def remove_tensor(data):
    if isinstance(data, dict):

        for key in list(data.keys()):

            if T.is_tensor(data[key]):
                try:
                    data[key] = data[key].item()
                except:
                    del data[key]

            elif isinstance(data[key], (tuple, list)) and len(data[key]) > 0:

                if T.is_tensor(data[key][0]):
                    del data[key]

                if isinstance(data[key][0], dict):
                    for x in data[key]:
                        remove_tensor(x)

            elif isinstance(data[key], dict):

                remove_tensor(data[key])


class TMSuperviseInferencer(TMSuperviseOperator, TMSupervisedEvaluatorMixin):
    """
    TMLearner
    """

    def __init__(self, hyper_params, machine, states=None):
        super().__init__(hyper_params, machine, scenario=TMScenario.Inference, states=states)

    def inference_batch(self, batch, scenario: TMScenario, local_rank):

        batch_traits = self.machine.BatchTraits

        batch = self.reallocate_data(batch, local_rank)
        batch_size = batch_traits.batch_size(batch)

        inferenced = self.machine.forward(batch, scenario=TMScenario.Inference)
        deep_merge_dict(inferenced, batch)
        yield inferenced

    def inference_channel(self, data_loader, channel: str, scenario: TMScenario,
                          local_rank: int):
        """

        Args:
            local_rank:
            data_loader:

        Returns:

        """
        P.OptimizerBehaviors.set_inference_mode(self.machine)

        with P.OptimizerBehaviors.no_grad():
            for i, batch in tqdm(enumerate(data_loader), desc=f"Inferencing Channel {channel}: "):
                try:

                    yield from self.inference_batch(batch, scenario, local_rank)

                except Exception as e:

                    batch_traits = self.machine.BatchTraits
                    shapes = batch_traits.shape(batch)
                    logger.error(f"Learn for {i}-th Batch, Data Shapes = {shapes}")
                    raise e

    def inference(self, local_rank, predict_batchstreams: TMDataStream, runtime_options):
        """

        Args:
            local_rank ():
            model ():
            optimization ():
            train_data_loader ():
            dev_data_loader ():
            test_data_loader ():
            problem_evaluator ():
            task_evaluator ():
            learn_conf ():

        Returns:

        """


        self.local_rank = local_rank

        self.distributed_strategy.init(local_rank)

        inference_machine_datastream = TMDataStream()
        inference_machine_datastream.level = TMDataLevel.Batch
        inference_machine_datastream.eval_channels = predict_batchstreams.eval_channels
        inference_machine_datastream.inference_channels = predict_batchstreams.inference_channels


        for channel in predict_batchstreams.inference_channels:

            inference_machine_datastream[channel] = self.inference_channel(predict_batchstreams[channel],
                                                                           channel=channel,
                                                                           scenario=TMScenario.Inference,
                                                                           local_rank=local_rank)
            # why cannot it step into the predict_channel function.

        for channel in predict_batchstreams.eval_channels:

            inference_machine_datastream[channel] = self.inference_channel(predict_batchstreams[channel],
                                                                           scenario=TMScenario.Evaluation,
                                                                           local_rank=local_rank)
            # why cannot it step into the predict_channel function.


        signal_returns = self.evaluate(inference_machine_datastream, local_rank, -1, -1)
        assert len(signal_returns) == 1
        evaluation_results = signal_returns[0][1]

        self.metric_logging_strategy.log(evaluation_results)


        return inference_machine_datastream

    def operate(self, machine_stream: TMDataStream, runtime_options):
        """

        Args:
            problem ():
            raw ():
            problem_modeler ():

        Returns:

        """

        self.runtime_options = runtime_options

        assert runtime_options.distributed.lower() != "ddp", "Inference does not support DDP"

        # logger.info(f"operate with parameter {runtime_options}")
        assert machine_stream.level == TMDataLevel.Batch, f"wrong data level, {machine_stream.level}"

        # memory_stream = self.fit_memory(machine_stream)
        # batch_stream = self.batchify(memory_stream)
        infered_batchstream = self.distributed_strategy.run(predict_worker, machine_stream, runtime_options)

        # infered_memory_stream = self.unbatchify(infered_batchstream)
        # infered_machine_stream = self.unfit_memory(infered_memory_stream)

        return infered_batchstream









