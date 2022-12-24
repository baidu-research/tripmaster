
from tripmaster.core.components.evaluator import MachineEvaluationResults, \
    EvaluationResults
from tripmaster.core.components.operator.strategies.metric_logging import TMMetricLoggingStrategy


import tableprint as tp
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


def print_evaluation_results(results: EvaluationResults, phrase):
    """

    Args:
        result:

    Returns:

    """
    if results.local_rank != 0:
        return

    headers = None
    rows = []
    for channel in results.performance.keys():
        problem_level_perf = results.performance[channel]
        if headers is None:
            headers = ["channel"] + list(problem_level_perf.keys())

        row = [f"{phrase}/{channel}",  ] + [scalar(problem_level_perf[k]) for k in headers[1:]]
        rows.append(row)

    if rows and results.local_rank == 0:
        tp.table(rows, headers, out=sys.stderr)

class TMTablePrintMetricLoggingStrategy(TMMetricLoggingStrategy):
    """
    TMTablePrintMetricLoggingStrategy
    """

    Name = "tableprint"

    def log_metrics(self, results, phase, with_loss=False):
        """

        Args:
            result:

        Returns:

        """
        if results.local_rank != 0:
            return
        headers = None
        rows = []
        for channel in results.performance.keys():
            perf = results.performance[channel]
            if headers is None:
                headers = ["channel"] + (["objective"] if with_loss else []) + list(perf.keys())
            start_idx = 2 if with_loss else 1
            row = [f"{phase}/{channel}"] + ([scalar(results.objective[channel])] if with_loss else []) + \
                  [scalar(perf[k]) for k in headers[start_idx:]]
            rows.append(row)

        if rows:
            tp.table(rows, headers, out=sys.stderr)

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


