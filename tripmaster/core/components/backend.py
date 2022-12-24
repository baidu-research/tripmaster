import abc
from typing import Type


class TMOptimizerBehaviors(abc.ABC):

    @classmethod
    @abc.abstractmethod
    def optimizer_state_dict(cls, optimizer):
        pass

    @classmethod
    @abc.abstractmethod
    def load_optimizer_state(cls, optimizer, state_dict):
        pass

    @classmethod
    @abc.abstractmethod
    def set_train_mode(self, machine):
        pass

    @classmethod
    @abc.abstractmethod
    def set_inference_mode(self, machine):
        pass

    @classmethod
    @abc.abstractmethod
    def no_grad(self):
        pass

    @classmethod
    @abc.abstractmethod
    def clear_grad(self, optimizer):
        pass

    @classmethod
    @abc.abstractmethod
    def backward_loss(self, loss):
        pass

    @classmethod
    @abc.abstractmethod
    def optimizer_step(self, optimizer):
        pass

    @classmethod
    @abc.abstractmethod
    def lrscheduler_step(self, lr_scheduler):
        pass

    @classmethod
    @abc.abstractmethod
    def clip_gradient_norm(self, parameters, clip_norm_threshold):
        pass 

    @classmethod
    @abc.abstractmethod
    def create_optimization_components(self, params,
                                   optimizer_type, optimizer_params,
                                   lr_scheduler_type, lr_scheduler_params,
                                   gradient_clip_val):
        pass


class TMTypes(object):

    Bool = None
    Int8 = None
    Int16 = None
    Int32 = None
    Int64 = None
    Float16 = None
    Float32 = None
    Float64 = None

class TMBasicTensorOperations(object):

    Tensor = None

    @classmethod
    def is_tensor(self, x):
        pass

    @classmethod
    def pad(self, *args, **kwargs):
        pass

    @classmethod
    def to_tensor(self, x, dtype=None):
        pass

    @classmethod
    def stack(self, list, dim):
        pass

    @classmethod
    def cast(self, x, type):
        pass

    @classmethod
    def to_numpy(self, x):
        pass

    @classmethod
    def to_device(self, x, device):
        pass

    @classmethod
    def device(cls, x):
        pass

    @classmethod
    def shape(self, x):
        pass

class TMBasicDeviceOperations(object):

    @classmethod
    def is_device(self, x):
        pass

    @classmethod
    def set_device(cls, device):
        pass

    @classmethod
    def cuda_available(cls):
        pass
    
    @classmethod
    @property
    def device_prefix(cls):
        pass 


class TMBasicModuleOperations(object):

    Module = None
    ModuleList = None
    ModuleDict = None

    @classmethod
    def load_state_dict(cls, module, state_dict):
        raise NotImplementedError()

    @classmethod
    def to_device(cls, module, device):
        raise NotImplementedError()


class TMBackend(abc.ABC):


    Name = None

    NoParallelStrategy = None
    DataParallelStrategy = None
    DistributedDataParallelStrategy = None

    OptimizationModule = None
    OptimizerBehaviors: TMOptimizerBehaviors = None

    DataLoader = None
    Dataset = None

    Sampler = None
    BatchSampler = None
    SubsetRandomSampler = None
    RandomSampler = None
    SequentialSampler = None
    DistributedSampler = None
    DistributedBatchSampler = None

    Types: TMTypes = None
    BasicTensorOperations: TMBasicTensorOperations = None
    BasicDeviceOperations: TMBasicDeviceOperations = None
    BasicModuleOperations: TMBasicModuleOperations = None


class TMBackendFactory(object):

    instance = None

    def __init__(self):

        self.name_backend_map = dict()
        self.chosen_backend = None

    def register(self, backend: Type[TMBackend]):

        self.name_backend_map[backend.Name] = backend

    def backends(self):

        for unit_uri in self.name_backend_map.keys():
            yield unit_uri

    def has_backend(self, name):
        return name in self.name_backend_map

    def choose(self, backend_name):
        if backend_name not in self.name_backend_map:
            raise Exception(f"unknown backend name {backend_name} ")

        if self.chosen_backend is not None:
            raise Exception("We donot allow dynamically changing backend")

        self.chosen_backend = backend_name

    def chosen(self) -> TMBackend:

        if self.chosen_backend is None:
            raise Exception("No backend is chosen")

        return self.name_backend_map[self.chosen_backend]

    @classmethod
    def get(cls):
        """

        Returns:

        """
        if cls.instance is None:
            cls.instance = TMBackendFactory()

        return cls.instance




