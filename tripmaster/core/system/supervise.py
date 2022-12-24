"""
TM Supervise System
"""

from tripmaster.core.concepts.data import TMDataStream
from tripmaster.core.system.system import TMSystem


class TMSuperviseSystem(TMSystem):
    """
    TMSuperviseSystem
    """

    def run(self, input_datastream: TMDataStream, runtime_options):
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

        input_datastream = self.build_machine_datastream(input_datastream)

        #        self.tp_modeler.update_downstream_component(self.task, self.problem)

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

            input_datastream = self.operator.fit_memory(input_datastream)
            input_datastream = self.operator.batchify(input_datastream)

            # CAUTION !!! do not delete this line !!!
            # We need the dummy_ref to make the system copied to child process in ddp
            self.operator.dummy_ref = self.evaluate_callback

            self.operator.evaluate_signal.connect(self.evaluate_callback)

            if self.is_learning():
                self.operator.good_model_discovered_signal.connect(
                    self.better_model_discovered)

            result = self.operator.operate(input_datastream, runtime_options)

            if not self.is_learning():  # result is the inferenced machine stream

                result = self.operator.unbatchify(result)
                result = self.operator.unfit_memory(result)
                result = self.recover_input_datastream(result)

            for callback in self.callbacks:
                callback.on_operation_finished(self)

        return result
