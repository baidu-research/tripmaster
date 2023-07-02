"""
TM system
"""
import abc
import weakref
from collections import defaultdict
from inspect import isclass
from typing import Optional, List

from tripmaster.core.components.environment.base import TMEnvironmentPool
from tripmaster.core.components.operator.operator import TMLearnerMixin
from tripmaster.core.concepts.component import TMSerializableComponent
from tripmaster.core.concepts.contract import TMContract, TMContractInterface
from tripmaster.core.concepts.data import TMDataLevel
from tripmaster.core.concepts.scenario import TMScenario
from tripmaster.core.system.system import TMSystem, TMSystemRuntimeCallbackInterface
from tripmaster.core.system.validation import TMSystemValidator
from tripmaster.utils.function import return_none
from tripmaster import logging
logger = logging.getLogger(__name__)

class TMReinforceSystem(TMSystem):
    """
    TMReinforceSystem
    """

    def run(self, env_pool_group, runtime_options):
        """

            Args:
                source:
                runtime_options:

            Returns:

        """
        self.pre_system_creation()

        self.lazy_update_test_setting()


        if runtime_options.mode == "data":
            self.post_system_creation()
            result = None
        else:

            if self.machine is None:
                self.lazy_build_machine_operator()

                self.post_system_creation()

                self.lazy_update_test_setting()

                if self.test_config is not None:
                    self.machine.test(self.test_config)
                    self.machine.validate = self.validate
                    self.operator.test(self.test_config)
                    self.operator.validate = self.validate

            self.operator.runtime(runtime_options)

            # CAUTION !!! do not delete this line !!!
            # We need the dummy_ref to make the system copied to child process in ddp
            self.operator.dummy_ref = self.evaluate_callback
            self.operator.evaluate_signal.connect(self.evaluate_callback)

            if self.is_learning():
                self.operator.good_model_discovered_signal.connect(
                    self.better_model_discovered)

            result = self.operator.operate(env_pool_group, runtime_options)

            for callback in self.callbacks:
                callback.on_operation_finished(self)

        return result

