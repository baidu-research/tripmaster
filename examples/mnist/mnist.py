import math

import numpy as np

from tripmaster.core.components.evaluator import TMMetricEvaluator
from tripmaster.core.components.loss import TMFunctionalLoss
from tripmaster.core.components.machine.machine import TMMachine
from tripmaster.core.components.problem import TMProblem
from tripmaster.core.components.task import TMTask
from tripmaster.core.concepts.contract import TMContractChannel

import paddle
from paddle import nn


from tripmaster.core.concepts.data import TMDataStream, TMDataLevel
from tripmaster.core.concepts.scenario import TMScenario
from tripmaster.core.launcher.launcher import launch
from tripmaster.core.system.supervise import TMSuperviseSystem
from tripmaster.utils.data import split_dataset
import random

class MnistDataStream(TMDataStream):

    def __init__(self, hyper_params, states=None):
        super().__init__(hyper_params, level=TMDataLevel.Task, states=states)

        if states is not None:
            self.load_states(states)
            return

        from paddle.vision.transforms import ToTensor
        train_dataset = [{"image": image, "label": label}
                         for image, label in paddle.vision.datasets.MNIST(mode='train', transform=ToTensor())]
        test_dataset = [{"image": image, "label": label}
                        for image, label in paddle.vision.datasets.MNIST(mode='test', transform=ToTensor())]

        random.shuffle(train_dataset)
        ratio = self.hyper_params.training_ratio
        train_dataset, dev_dataset = split_dataset(train_dataset, [ratio, 1 - ratio])

        self["train"] = train_dataset
        self["dev"] = dev_dataset
        self["test"] = test_dataset

        self.learn_channels = ['train']
        self.eval_channels = ['dev', 'test']
        self.inference_channels = ['test']

class ImageClassificationTask(TMTask):

    ForwardProvisionSchema = {
        TMContractChannel.Source: {"image": object},
        TMContractChannel.Target: {"label": int}
    }
    BackwardRequestSchema = {
        TMContractChannel.Inference: {"inference_label": int}
    }

    Evaluator = None

class TensorClassificationProblem(TMProblem):

    ForwardProvisionSchema = {
        TMContractChannel.Source: {"tensor": np.ndarray},
        TMContractChannel.Target: {"label": int}
    }
    BackwardRequestSchema = {
        TMContractChannel.Inference: {"inference_label": int}
    }

    Evaluator = None


class ClassificationEvaluator(TMMetricEvaluator):

    Metrics = {"label": [paddle.metric.Precision(), paddle.metric.Recall()]}

class ClassificationLoss(TMFunctionalLoss):

    Func = paddle.nn.functional.cross_entropy
    LearnedFields = ["logit"]
    TruthFields = ["label"]

class Tensor2DClassificationMachine(TMMachine):
    ForwardRequestSchema = {
        TMContractChannel.Source: {"tensor": paddle.Tensor},
        TMContractChannel.Target: {"label": int}
    }
    BackwardProvisionSchema = {
        TMContractChannel.Learn: {"logit": paddle.Tensor},
        TMContractChannel.Inference: {"inference_label": int}
    }

    Loss = ClassificationLoss
    Evaluator = ClassificationEvaluator
    EvaluatorInferenceContract = {"inference_label": "label"}

    def __init__(self, hyper_params,  states=None):
        super().__init__(hyper_params, states=states)

        self.conv1 = nn.Conv2D(1, self.arch_params.channel1, self.arch_params.conv_kernel, 1)
        conv1_out_h = self.arch_params.image_size[0] - (self.arch_params.conv_kernel - 1)
        conv1_out_w = self.arch_params.image_size[1] - (self.arch_params.conv_kernel - 1)

        self.conv2 = nn.Conv2D(self.arch_params.channel1, self.arch_params.channel2,
                               self.arch_params.conv_kernel, 1)
        conv2_out_h = conv1_out_h - (self.arch_params.conv_kernel - 1)
        conv2_out_w = conv1_out_w - (self.arch_params.conv_kernel - 1)

        self.pool = nn.MaxPool2D(self.arch_params.pool_kernel)

        pool_out_h = math.floor((conv2_out_h - (self.arch_params.pool_kernel - 1)) / self.arch_params.pool_kernel + 1)
        pool_out_w = math.floor((conv2_out_w - (self.arch_params.pool_kernel - 1)) / self.arch_params.pool_kernel + 1)

        self.dropout1 = nn.Dropout(self.arch_params.dropout1)
        self.dropout2 = nn.Dropout(self.arch_params.dropout2)

        self.fc1 = nn.Linear(self.arch_params.channel2 * pool_out_h * pool_out_w, self.arch_params.ff_dim)
        self.fc2 = nn.Linear(self.arch_params.ff_dim, self.arch_params.class_num)

        if states:
            self.load_states(states)

    def forward(self, inputs, scenario=None):

        x = inputs["tensor"]
        x = self.conv1(x)
        x = paddle.nn.functional.relu(x)
        x = self.conv2(x)
        x = paddle.nn.functional.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)
        x = paddle.flatten(x, 1)
        x = self.fc1(x)
        x = paddle.nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        results = dict()
        if scenario in {TMScenario.Learning, TMScenario.Evaluation}:
            results["logit"] = x
        if scenario in {TMScenario.Evaluation, TMScenario.Inference}:
            results["inference_label"] = paddle.argmax(x, axis=-1)

        return results

from tripmaster.core.components.operator.supervise import TMSuperviseLearner


class MnistLearner(TMSuperviseLearner):

    Optimizer = paddle.optimizer.Adam
    LRScheduler = paddle.optimizer.lr.ExponentialDecay

from tripmaster.core.components.operator.supervise import TMSuperviseInferencer
from tripmaster.core.system.system import TMSystem

class ImageClassificationSystem(TMSuperviseSystem):

    TaskType = ImageClassificationTask
    Task2ProblemContract = {"image": "tensor"}
    ProblemType = TensorClassificationProblem
    MachineType = Tensor2DClassificationMachine

class ImageClassificationLearningSystem(ImageClassificationSystem):

    OperatorType = MnistLearner


class ImageClassificationInferenceSystem(ImageClassificationSystem):

    OperatorType = TMSuperviseInferencer



from tripmaster.core.app.standalone import TMStandaloneApp

from tripmaster.core.app.io import TMOfflineInputStream

class MnistInputStream(TMOfflineInputStream):

    DataStreamType = MnistDataStream

class MnistLearningApplication(TMStandaloneApp):
    InputStreamType = MnistInputStream
    SystemType = ImageClassificationLearningSystem

from tripmaster.core.app.io import TMOfflineOutputStream

class MnistInferenceApplication(TMStandaloneApp):
    InputStreamType = MnistInputStream
    OutputStreamType = TMOfflineOutputStream
    SystemType = ImageClassificationInferenceSystem

import click

@click.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.argument("operator", type=click.Choice(("learning", "inference")),
                default="learning")
def run(operator):
    if operator == "learning":
        launch(MnistLearningApplication)
    else: # operator == "inference":
        launch(MnistInferenceApplication)

if __name__ == "__main__":
    run()
