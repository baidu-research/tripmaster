"""
application
"""
import yaml
from omegaconf import OmegaConf

from tripmaster.core.app.config import TMConfig
from tripmaster.core.concepts.component import TMConfigurable
from tripmaster.core.concepts.hyper_params import TMHyperParams
from tripmaster.core.system.system import TMSystemRuntimeCallbackInterface, is_multi_system, to_save
from tripmaster import logging
import os

logger = logging.getLogger(__name__)

class TMDefaultSystemRuntimeCallback(TMSystemRuntimeCallbackInterface, TMConfigurable):
    """
    TMDefaultSystemRuntimeCallback
    """

    def on_task_data_loaded(self, task_data):
        if not self.hyper_params.io.input or not self.hyper_params.io.input.task:
            return

        task_serialize_config = self.hyper_params.io.input.task.serialize

        if task_serialize_config and task_serialize_config.save:
            logger.info(f"Saving task data with serialize config {self.hyper_params.io.input.task.serialize}")
            task_data.serialize(self.hyper_params.io.input.task.serialize.path)

    def on_problem_data_built(self, problem_data):
        if not self.hyper_params.io.input or not self.hyper_params.io.input.problem:
            return
        problem_serialize_config = self.hyper_params.io.input.problem.serialize
        if problem_serialize_config and problem_serialize_config.save:
            logger.info(f"Saving problem data with serialize config {self.hyper_params.io.input.problem.serialize}")
            problem_data.serialize(self.hyper_params.io.input.problem.serialize.path)

    def on_machine_data_built(self, machine_data):
        if not self.hyper_params.io.input or not self.hyper_params.io.input.machine:
            return
        machine_serialize_config = self.hyper_params.io.input.machine.serialize
        if machine_serialize_config and machine_serialize_config.save:
            logger.info(f"Saving machine data with serialize config {self.hyper_params.io.input.machine.serialize}")
            machine_data.serialize(self.hyper_params.io.input.machine.serialize.path)

    def on_data_phase_finished(self, system):

        if to_save(system.hyper_params.task):
            system.task.serialize(system.hyper_params.task.serialize.path)
            logger.info("task serialized")

        if to_save(system.hyper_params.tp_modeler):
            system.tp_modeler.serialize(system.hyper_params.tp_modeler.serialize.path)
            logger.info("tp_modeler serialized")

        if to_save(system.hyper_params.problem):
            system.problem.serialize(system.hyper_params.problem.serialize.path)
            logger.info("problem serialized")

        if to_save(system.hyper_params.pm_modeler):
            system.pm_modeler.serialize(system.hyper_params.pm_modeler.serialize.path)
            logger.info("pm_modeler serialized")



class TMStandaloneApp(TMConfigurable):

    InputStreamType = None
    OutputStreamType = None
    SystemType = None

    def __init__(self, hyper_params, callbacks=None):
        super().__init__(hyper_params)

        self.callbacks = callbacks

        if self.callbacks is None:
            self.callbacks = [TMDefaultSystemRuntimeCallback(self.hyper_params)]

        self.input_stream = self.InputStreamType(self.hyper_params.io.input)

        # self.output_stream = self.OutputStreamType(self.hyper_params.io.output)

        self.input_data_stream = self.input_stream.data_stream()

        if self.OutputStreamType:
            self.output_data_stream = self.OutputStreamType(self.hyper_params.io.output)

        if is_multi_system(self.SystemType):
            system_hyper_param = self.hyper_params.multisystem
        else:
            system_hyper_param = self.hyper_params.system
        self.system = self.SystemType.create(system_hyper_param,
                                             callbacks=self.callbacks)

        if self.hyper_params.operator == "from_scratch":
            self.system.operator_from_scratch()


    @classmethod
    def check_system_contracts(cls):

        cls.SystemType.check_contracts()

    @classmethod
    def generate_conf_template(cls):

        conf = TMConfig.default()
        conf = OmegaConf.to_container(conf)

        print(yaml.safe_dump(conf))

    @classmethod
    def parse_hyper_parameters(cls, conf_file_path, cmd_args=None):

        from omegaconf import OmegaConf, open_dict
        cmd_args = cmd_args if cmd_args is not None else []

        base_conf = TMConfig.default()
        
        cli_conf = OmegaConf.from_cli(cmd_args)

        assert isinstance(conf_file_path, (list, tuple)) and isinstance(conf_file_path[0], str)
        with open_dict(base_conf), open_dict(cli_conf):
            conf = base_conf
            for user_conf_path in conf_file_path:
                user_conf = OmegaConf.load(user_conf_path)
                with open_dict(user_conf):
                    conf.merge_with(user_conf)

            conf.merge_with(cli_conf)

        if not conf.job.startup_path:
            conf.job.startup_path = os.getcwd()

        conf = TMHyperParams(OmegaConf.to_container(conf, resolve=True))

#        conf.freeze()

        return conf 

    def test(self, test_config):
        logger.info(f"the application is running in test mode with test setting {test_config}")
        self.system.test(test_config)
        if test_config.sample_num > 0:
            self.input_data_stream.test(TMHyperParams(sample_num=test_config.sample_num))

    def run(self):

        runtime_options = self.hyper_params.job

        self.input_data_stream.add_sampled_training_eval_channels()

        result = self.system.run(self.input_data_stream, runtime_options)

        if not self.system.is_learning() and self.output_data_stream is not None:
            self.output_data_stream.write(result)
            