import inspect
from pkgutil import iter_modules
from pathlib import Path
from importlib import import_module

from tripmaster import logging

logger = logging.getLogger(__name__)

def load_plugins(plugin_name, plugin_init_path, callbacks):

    package_dir = Path(plugin_init_path).resolve().parent
    for (_, module_name, _) in iter_modules([package_dir]):

        module_path = f"{plugin_name}.{module_name}"
        # import the module and iterate through its attributes
        try:
            module = import_module(module_path)
        except Exception as e:
            logger.warning(f"unable to import plugin {module_path}: {e}")
            logger.exception(e)
            # continue

        for attribute_name in dir(module):
            attribute = getattr(module, attribute_name)

            if inspect.isclass(attribute):
                for type, func in callbacks.items():
                    if issubclass(attribute, type):
                        func(attribute)

