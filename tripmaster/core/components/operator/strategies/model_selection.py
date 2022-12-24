import abc
from dataclasses import dataclass
from typing import Type
import operator

from tripmaster.core.concepts.component import TMConfigurable, TMSerializableComponent

from tripmaster import logging


logger = logging.getLogger(__name__)

import weakref

class TMModelSelectionStrategy(TMSerializableComponent):
    """
    TMModelSelectionStrategy
    """
    Name: str = None


    @abc.abstractmethod
    def select_model(self, results, machine, learner):

        pass


@dataclass
class BestOneModelSelectorConfig:
    stage: str = "problem"
    channel: str = ""
    metric: str = ""
    better: str = "max"
    save_prefix: str = "best_model"


@dataclass
class SelectedModelInfo:
    stage: str = "problem"
    channel: str = ""
    metric: str = ""
    perf: float = 0
    epoch: int = 0
    step: int = 0
    prefix: str = ""
    id_patten: str = "{prefix}-{stage}-{channel}-{metric}-{perf}@{epoch}.{step}"

@dataclass
class HotpotSelectedModelInfo(SelectedModelInfo):
    hotpot: bool = True
    cur_best: bool = False


class BestOneModelSelectionStrategy(TMModelSelectionStrategy):
    """
    TMModelSelectionStrategy
    """

    Name = "best_one"

    def __init__(self, hyper_params):

        super().__init__(hyper_params)

        assert self.hyper_params.better in {"max", "min"}
        assert self.hyper_params.stage in {"task", "problem", "machine"}

        import operator

        if self.hyper_params.better == "max":
            self.better_func = operator.gt
            self.cur_best_perf = float("-inf")
        else:
            self.better_func = operator.lt
            self.cur_best_perf = float("inf")

        self.best_model_path = None

        self.prev_model_name = None

#        signal(TMEvaluationSignals.ON_PROBLEM_STREAM_EVALUATED).connect(self)
#        signal(TMEvaluationSignals.ON_TASK_STREAM_EVALUATED).connect(self)

    def select_model(self, results, machine, learner):

        result = results[self.hyper_params.stage]


        perf = result.performance

        channel = self.hyper_params.channel
        if channel not in perf:
            raise Exception(f"the channel {channel} not in evaluation metrics {perf.keys()}")

        perf = perf[self.hyper_params.channel]

        metric = self.hyper_params.metric
        if self.hyper_params.metric not in perf:
            raise Exception(f"the metric {metric} not in evaluation metrics {perf.keys()}")

        if result.local_rank == 0:
            import os
            if self.better_func(perf[metric], self.cur_best_perf):

                if self.prev_model_name:
                    os.remove(self.prev_model_name + ".model.pt")
                #    os.remove(self.prev_model_name + ".trainer.pt")

                self.cur_best_perf = perf[metric]

                logger.warning(f"better model found with performance {metric} = {self.cur_best_perf}, saving... ")

                model_info = SelectedModelInfo(
                    prefix=self.hyper_params.prefix,
                    stage=self.hyper_params.stage,
                    metric=self.hyper_params.metric,
                    channel=self.hyper_params.channel,
                    perf=self.cur_best_perf,
                    epoch=result.epoch,
                    step=result.step,
                )
                if self.hyper_params.id_pattern:
                    model_info.id_patten = self.hyper_params.id_pattern

                file_name = model_info.id_patten.format(prefix=model_info.prefix, stage=model_info.stage,
                                                        channel=model_info.channel,
                                             metric=model_info.metric, perf=model_info.perf,
                                             epoch=model_info.epoch, step=model_info.step)

                self.best_model_path = file_name + ".model.pt"
                machine.serialize(path=self.best_model_path)
                yield model_info
#                file_name = f"{self.hyper_params.save_prefix}-epoch={result.epoch}-{metric}={self.cur_best_perf}"


                #torch.save({
                #    'state_dict': self.learner.machine.state_dict(),
                #    TMMachine.HYPER_PARAMS_KEY: self.learner.machine.hyper_params
                #}, file_name + ".model.pt")

#                torch.save({

#                    'optimization': trainer.optimization.state_dict(),
#                    'lr_sheduler': self.lr_scheduler.state_dict(),
#                    'best_perf': {metric: self.cur_best_perf}
#                }, file_name + ".trainer.pt")

                # self.prev_model_name = file_name


class MultiMetricModelSelectionStrategy(TMModelSelectionStrategy):
    """
    TMModelSelectionStrategy
    """

    Name = "multi_metric"

    def __init__(self, hyper_params):

        super().__init__(hyper_params)
        assert isinstance(self.hyper_params.better, list), "multi metric selector needs better as list"

        assert self.hyper_params.stage in {"task", "problem", "machine"}

        self.better = self.hyper_params.better
        self.best_model_path = None
        self.better_func = []
        self.cur_best_perf = []

        self.prev_model_name = {}
        for better in self.hyper_params.better:
            assert better in {"max", "min"}

            if better == "max":
                self.better_func.append(operator.gt)
                self.cur_best_perf.append(float("-inf"))
            else:
                self.better_func.append(operator.lt)
                self.cur_best_perf.append(float("inf"))


#        signal(TMEvaluationSignals.ON_PROBLEM_STREAM_EVALUATED).connect(self)
#        signal(TMEvaluationSignals.ON_TASK_STREAM_EVALUATED).connect(self)

    def select_model(self, results, machine, learner):

        result = results[self.hyper_params.stage]

        perf = result.performance

        channel = self.hyper_params.channel
        if channel not in perf:
            raise Exception(f"the channel {channel} not in evaluation metrics {perf.keys()}")

        perf = perf[self.hyper_params.channel]

        metrics = self.hyper_params.metric

        for i, metric in enumerate(metrics):
            if result.local_rank == 0:
                if metric not in perf:
                    raise Exception(f"the metric {metric} not in evaluation metrics {perf.keys()}")


                import os
                better_func = self.better_func[i]
                cur_best_perf = self.cur_best_perf[i]
                if better_func(perf[metric], cur_best_perf):

                    if metric in self.prev_model_name:
                        os.remove(self.prev_model_name[metric] + ".model.pt")
                    #    os.remove(self.prev_model_name + ".trainer.pt")

                    self.cur_best_perf[i] = perf[metric]

                    logger.warning(f"better model found with performance {metric} = {self.cur_best_perf[i]}, saving... ")

                    # file_name = f"{self.hyper_params.prefix}-epoch={result.epoch}-{metric}={self.cur_best_perf[i]}"
                    #

                    #torch.save({
                    #    'state_dict': self.learner.machine.state_dict(),
                    #    TMMachine.HYPER_PARAMS_KEY: self.learner.machine.hyper_params
                    #}, file_name + ".model.pt")

    #                torch.save({

    #                    'optimization': trainer.optimization.state_dict(),
    #                    'lr_sheduler': self.lr_scheduler.state_dict(),
    #                    'best_perf': {metric: self.cur_best_perf}
    #                }, file_name + ".trainer.pt")


                    model_info = SelectedModelInfo(
                        prefix=self.hyper_params.prefix,
                        stage=self.hyper_params.stage,
                        channel=self.hyper_params.channel,
                        metric=self.hyper_params.metric,
                        perf=perf[metric],
                        epoch=result.epoch,
                        step=result.step,
                    )
                    if self.hyper_params.id_pattern:
                        model_info.id_patten = self.hyper_params.id_pattern


                    file_name = model_info.id_patten.format(prefix=model_info.prefix, stage=model_info.stage,

                                                            channel=model_info.channel,
                                                            metric=model_info.metric, perf=model_info.perf,
                                                            epoch=model_info.epoch, step=model_info.step)

                    self.best_model_path = file_name + ".model.pt"
                    machine.serialize(path=self.best_model_path)

                    self.prev_model_name[metric] = file_name

                    yield metric, model_info


class ModelHotpotMultiMetricModelSelectionStrategy(TMModelSelectionStrategy):
    """
    TMModelSelectionStrategy
    """

    Name = "model_hotpot_multi_metric"

    def __init__(self, hyper_params):

        super().__init__(hyper_params)
        assert isinstance(self.hyper_params.better, list), "multi metric selector needs better as list"

        assert self.hyper_params.stage in {"task", "problem", "machine"}

        self.better = self.hyper_params.better
        self.best_model_path = None
        self.better_func = []
        self.cur_best_perf = []
        self.metric_clip = []
        for m in self.hyper_params.metric_clip:
            self.metric_clip.append(m)
        # self.prev_model_name = {}
        for better in self.hyper_params.better:
            assert better in {"max", "min"}

            if better == "max":
                self.better_func.append(operator.gt)
                self.cur_best_perf.append(float("-inf"))
            else:
                self.better_func.append(operator.lt)
                self.cur_best_perf.append(float("inf"))

#        signal(TMEvaluationSignals.ON_PROBLEM_STREAM_EVALUATED).connect(self)
#        signal(TMEvaluationSignals.ON_TASK_STREAM_EVALUATED).connect(self)

    def select_model(self, results, machine, learner):

        result = results[self.hyper_params.stage]

        perf = result.performance

        channel = self.hyper_params.channel
        if channel not in perf:
            raise Exception(f"the channel {channel} not in evaluation metrics {perf.keys()}")

        perf = perf[self.hyper_params.channel]

        metrics = self.hyper_params.metric

        for i, metric in enumerate(metrics):
            if result.local_rank == 0:
                if metric not in perf:
                    raise Exception(f"the metric {metric} not in evaluation metrics {perf.keys()}")


                import os
                better_func = self.better_func[i]
                cur_best_perf = self.cur_best_perf[i]

                if better_func(perf[metric], cur_best_perf):
                    self.cur_best_perf[i] = perf[metric]
                    is_cur_best = True
                else:
                    is_cur_best = False
                if better_func(perf[metric], self.metric_clip[i]):
                    hotpot = True
                else:
                    hotpot = False
                if hotpot or is_cur_best:

                    # if metric in self.prev_model_name:
                    #     os.remove(self.prev_model_name[metric] + ".model.pt")
                    #    os.remove(self.prev_model_name + ".trainer.pt")

                    # self.cur_best_perf[i] = perf[metric]

                    logger.warning(f"Model founded with performance {metric} = {perf[metric]}, saving... ")

                    # file_name = f"{self.hyper_params.prefix}-epoch={result.epoch}-{metric}={self.cur_best_perf[i]}"
                    #

                    #torch.save({
                    #    'state_dict': self.learner.machine.state_dict(),
                    #    TMMachine.HYPER_PARAMS_KEY: self.learner.machine.hyper_params
                    #}, file_name + ".model.pt")

    #                torch.save({

    #                    'optimization': trainer.optimization.state_dict(),
    #                    'lr_sheduler': self.lr_scheduler.state_dict(),
    #                    'best_perf': {metric: self.cur_best_perf}
    #                }, file_name + ".trainer.pt")


                    model_info = HotpotSelectedModelInfo(
                        hotpot=hotpot,
                        prefix=self.hyper_params.prefix,
                        stage=self.hyper_params.stage,
                        channel=self.hyper_params.channel,
                        metric=self.hyper_params.metric,
                        perf=perf[metric],
                        epoch=result.epoch,
                        step=result.step,
                        cur_best=is_cur_best,
                    )
                    if self.hyper_params.id_pattern:
                        model_info.id_patten = self.hyper_params.id_pattern


                    file_name = model_info.id_patten.format(prefix=model_info.prefix, stage=model_info.stage,
                                                            channel=model_info.channel,
                                                            metric=model_info.metric, perf=model_info.perf,
                                                            epoch=model_info.epoch, step=model_info.step)

                    # self.best_model_path = file_name + ".model.pt"
                    # machine.serialize(path=self.best_model_path)

                    # self.prev_model_name[metric] = file_name

                    yield metric, model_info


class TMModelSelectionStrategyFactory(object):

    instance = None

    def __init__(self):

        self.name_strategy_map = dict()

    def register(self, strategy: Type[TMModelSelectionStrategy]):

        self.name_strategy_map[strategy.Name] = strategy

    def strategies(self):

        for name in self.name_strategy_map.keys():
            yield name

    def choose(self, name) -> TMModelSelectionStrategy:

        return self.name_strategy_map[name]

    @classmethod
    def get(cls):
        """

        Returns:

        """
        if cls.instance is None:
            cls.instance = TMModelSelectionStrategyFactory()

        return cls.instance