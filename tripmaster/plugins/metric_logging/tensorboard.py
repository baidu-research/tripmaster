from typing import Union
from tripmaster.core.components.backend import TMBackendFactory

P = TMBackendFactory.get().chosen()
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.simplefilter("ignore", category=PendingDeprecationWarning)

if P.Name == "torch":
    from torch.utils.tensorboard import SummaryWriter as Writer
elif P.Name == "paddle":
    from visualdl import LogWriter as Writer 
else:
    raise Exception("Unkonwn backend {P.Name}")

from tripmaster.core.components.evaluator import MachineEvaluationResults, EvaluationResults
from tripmaster.core.components.operator.strategies.metric_logging import TMMetricLoggingStrategy

import sys


def scalar(x):
    """

    Args:
        x:

    Returns:

    """

    from tripmaster import TMBackendFactory
    B = TMBackendFactory.get().chosen()

    if B.BasicTensorOperations.is_tensor(x):
        return x.item()
    else:
        return x


class TMTensorboardMetricLoggingStrategy(TMMetricLoggingStrategy):
    """
    TMTablePrintMetricLoggingStrategy
    """

    Name = "tensorboard"

    def text(self, key, text, step):
        """

        Args:
            x:

        Returns:

        """
        tb_summary_writer = Writer(self.hyper_params.path)
        tb_summary_writer.add_text(key, text, global_step=step)

    def scalar(self, key, scalar, step):
        """

        Args:
            scalar:
            step:

        Returns:

        """
        tb_logger = Writer(self.hyper_params.path)
        tb_logger.add_scalar(key, scalar, step)

    def log_metrics(self, results: EvaluationResults, phase, with_loss=False):
        """

        Args:
            result:

        Returns:

        """

        if results.local_rank != 0:
            return

        metric_logger = Writer(self.hyper_params.path)

        for channel in results.performance.keys():
            perf = results.performance[channel]
            if with_loss:
                metric_logger.add_scalar(f"{phase}/{channel}/objective", scalar(results.objective[channel]), results.step)
            for key in perf:
                metric_logger.add_scalar(f"{phase}/{channel}/{key}", scalar(perf[key]), results.step)

    def log(self, evaluation_results):
        """

        Args:
            evaluation_results:

        Returns:

        """
        if "machine" in evaluation_results:
            self.log_metrics(evaluation_results["machine"], "machine", with_loss=True)

        if "problem" in evaluation_results:
            self.log_metrics(evaluation_results["problem"], "problem")

        if "task" in evaluation_results:
            self.log_metrics(evaluation_results["task"], "task")