
from tripmaster.core.components.backend import TMBackend, TMBackendFactory

from tripmaster.plugins.load import load_plugins

register_funcs = {TMBackend: TMBackendFactory.get().register}

load_plugins(__name__, __file__, register_funcs)
#
# package_dir = Path(__file__).resolve().parent
# for (_, module_name, _) in iter_modules([package_dir]):
#
#     module_path = f"{__name__}.{module_name}"
#     # import the module and iterate through its attributes
#     try:
#         module = import_module(module_path)
#     except Exception as e:
#         logger.warning(f"unable to import plugin {module_path}")
#         continue
#
#     for attribute_name in dir(module):
#         attribute = getattr(module, attribute_name)
#
#         if inspect.isclass(attribute) and issubclass(attribute, TMBackend):
#             # Add the class to this package's variables
#             TMBackendFactory.get().register(attribute)

import os
if "TM_BACKEND" in os.environ:
    backend_name = os.environ["TM_BACKEND"]
    if not TMBackendFactory.get().has_backend(backend_name):
        raise Exception(f"Unknown backend {backend_name}, set by environ variable TM_BACKEND")

    TMBackendFactory.get().choose(backend_name)
else:
    TMBackendFactory.get().choose("paddle")