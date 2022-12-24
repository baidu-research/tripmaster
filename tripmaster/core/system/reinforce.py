"""
TM system
"""
from tripmaster.core.components.environment.base import TMEnvironmentPool
from tripmaster.core.concepts.data import TMDataLevel
from tripmaster.core.concepts.scenario import TMScenario
from tripmaster.core.system.system import TMSystem


class TMReinforceSystem(TMSystem):
    """
    TMReinforceSystem
    """

    def run(self, env_pool: TMEnvironmentPool, runtime_options):
        """
        Args:
            input_data_stream: None, TMLearningTaskData, TMInferenceTaskData,
                            TMLearningProblemData, TMInferenceProblemData.
                            If not none, it is loaded by application from the serialization options
            resource:

        Returns:

        """

        self.pre_system_creation()

        self.build_data_pipeline()

        self.lazy_update_test_setting()

        if self.is_learning():
            scenario = TMScenario.Learning
        else:
            scenario = TMScenario.Inference

        env_pool.test(self.test_config)

        if env_pool.level == TMDataLevel.Task:
            if self.tp_modeler is not None:
                env_pool = env_pool.apply_modeler(self.tp_modeler, scenario=scenario)
            else:
                env_pool.level = TMDataLevel.Problem

        if env_pool.level == TMDataLevel.Problem:
            if self.pm_modeler is not None:
                env_pool = env_pool.apply_modeler(self.pm_modeler, scenario=scenario)
            else:
                env_pool.level = TMDataLevel.Machine

        for callback in self.callbacks:
            callback.on_data_phase_finished(self)

        if runtime_options.data_mode:
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

            result = self.operator.operate(env_pool, runtime_options)

            if not self.is_learning():  # result is the inferenced machine stream

                # result = self.operator.unbatchify(result)
                # result = self.operator.unfit_memory(result)

                result = self.recover_input_datastream(result)

            for callback in self.callbacks:
                callback.on_operation_finished(self)

        return result
