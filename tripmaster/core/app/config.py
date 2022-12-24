"""
config entries
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List, Union

from omegaconf import OmegaConf
import os

@dataclass
class Serializable:
    """
    ProblemModeler
    """
    save: bool = False
    load: bool = False
    path: Optional[str] = None
    multi_path: Optional[Dict] = None


@dataclass
class Strategy:
    type: str = ""
    strategies: Dict[str, Any] = field(default_factory=lambda: dict())


@dataclass
class Component(dict):
    """
    ProblemModeler
    """
    serialize: Optional[Serializable] = None

# @dataclass
# class Contract:
#     """
#     contracts
#     """
#     task2modeler: Dict[str, str] = field(default_factory=lambda: {})
#     modeler2problem: Dict[str, str] = field(default_factory=lambda: {})
#     problem4machine: Dict[str, str] = field(default_factory=lambda: {})
#     problem2evaluator: Dict[str, str] = field(default_factory=lambda: {})
#     problem2loss: Dict[str, str] = field(default_factory=lambda: {})

@dataclass
class Machine:
    """
    machine configurations
    """

    arch: Optional[Dict[str, Any]] = None
    loss: Optional[Dict[str, Any]] = None
    evaluator: Optional[Dict[str, Any]] = None


@dataclass
class MultiMachine:
    """
    machine configurations
    """

    weights: Dict[str, float] = field(default_factory=lambda: dict())

@dataclass
class Task:
    """
    Task configurations
    """
    evaluator: Optional[Component] = None


@dataclass
class Problem:
    """
    problem configurations
    """
    evaluator: Optional[Component] = None


@dataclass
class DataLoader(dict):
    worker_num: int = 2
    pip_memory: bool = False
    timeout: Optional[int] = None
    resource_allocation_range: int = 10000
    drop_last: bool = False


@dataclass
class Optimizer:

    strategy: Dict[str, Any] = field(default_factory=lambda: dict())
    algorithm: Dict[str, Any] = field(default_factory=lambda: dict())
    lr_scheduler: Dict[str, Any] = field(default_factory=lambda: dict())
    gradient_clip_val: Optional[float] = None
    epochs: int = 0


@dataclass
class ComputingResource:
    gpus: int = 1
    cpus: int = 4
    gpu_per_trial: int = 1
    cpu_per_trial: int = 4

@dataclass
class MemoryResource:

    learning_memory_limit: int = 0
    inferencing_memory_limit: int = 0


@dataclass
class Resource:

    computing: ComputingResource = ComputingResource()
    memory: Optional[MemoryResource] = None

@dataclass
class Operator:
    detect_anomaly: bool = False


@dataclass
class Learner(Operator):

    optimizer: Optimizer = field(default_factory=lambda: Optimizer())
    modelselector: Dict[str, Any] = field(default_factory=lambda: dict())
    evaluator_trigger: Dict[str, Any] = field(default_factory=lambda: dict())


@dataclass
class Inferencer(Operator):
    pass


@dataclass
class Repo(dict):

    server: str = ""
    data_dir: str = ""


@dataclass
class System(Component):
    """
    AppConfig
    """


    task: Optional[Task] = None
    tp_modeler: Optional[Dict] = None
    problem: Optional[Problem] = None
    pm_modeler: Optional[Dict] = None
    machine: Machine = field(default_factory=lambda: Machine())
    learner: Optional[Learner] = None
    inferencer: Optional[Inferencer] = None

#    contracts: Contracts = Contracts()


@dataclass
class MultiSystem(Component):
    """
    AppConfig
    """

    subsystems:  Dict[str, str] = field(default_factory=lambda: dict())
    machine: MultiMachine = field(default_factory=lambda: MultiMachine())
    learner: Optional[Learner] = None
    inferencer: Optional[Inferencer] = None

@dataclass
class Data:

    task: Optional[Serializable] = None
    problem: Optional[Serializable] = None
    machine: Optional[Serializable] = None
    train_eval_sampling_ratio: float = 0


@dataclass
class IO:
    env: Optional[Dict[str, Any]] = None
    input: Optional[Dict[str, Any]] = None
    output: Optional[Dict[str, Any]] = None


@dataclass
class Test:
    """
    test mode configurations
    """
    sample_num: int = 16
    epoch_num: int = 3
    validate: bool = False


@dataclass
class Job:
    """
    Job configurations
    """
    startup_path: Optional[str] = None
    output_path: Optional[str] = None

    ray_tune: bool = False

    testing: bool = False
    test: Optional[Test] = None

    data_mode: bool = False

    # operation:
    #   from_scratch: reinitialize the machine and operator;
    #   continue: continue operation on existing machine and operation states
    operation: str = "continue"

    batching: Strategy = Strategy()
    dataloader: Optional[DataLoader] = None

    resource: Resource = Resource()
    distributed: str = "no"

    metric_logging: Strategy = Strategy()




@dataclass
class App:

    repo: Optional[Repo] = None
    system: Optional[System] = None # field(default_factory=lambda: System())
    multisystem: Optional[MultiSystem] = None
    io: IO = field(default_factory=lambda: IO())
    job: Job = field(default_factory=lambda: Job())

    launcher: Strategy = Strategy()


class TMConfig(object):
    """
    TM config
    """

    conf = None

    @staticmethod
    def load(yaml_file_path):
        """

        Args:
            yaml_file_path ():

        Returns:

        """

        schema = OmegaConf.structured(App)
        conf = OmegaConf.load(yaml_file_path)

        if not conf.job.startup_path:
            conf.job.startup_path = os.getcwd()

        TMConfig.conf = OmegaConf.merge(schema, conf)
#        TMConfig.conf = addict.Dict(OmegaConf.to_container(conf, resolve=True))

        return TMConfig.conf

    @staticmethod
    def default():
        """

        Returns:

        """
        return OmegaConf.structured(App)

    @staticmethod
    def get():
        """

        Returns:

        """

        if TMConfig.conf is None:

            default = OmegaConf.structured(App)
            default.job.startup_path = os.getcwd()

            TMConfig.conf = default

        return TMConfig.conf

    @staticmethod
    def set(conf):
        """

        Returns:

        """
        TMConfig.conf = conf