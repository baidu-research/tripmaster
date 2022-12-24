import tripmaster.utils.logging as logging
from . import plugins

from tripmaster.core.components.backend import TMBackendFactory
P = TMBackendFactory.get().chosen()
T = P.BasicTensorOperations
M = P.BasicModuleOperations
D = P.BasicDeviceOperations