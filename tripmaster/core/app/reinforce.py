"""
reinforce app
"""
import os

import yaml
from omegaconf import OmegaConf

from tripmaster.core.app.config import TMConfig
from tripmaster.core.app.standalone import TMDefaultSystemRuntimeCallback
from tripmaster.core.concepts.component import TMConfigurable
from tripmaster.core.concepts.hyper_params import TMHyperParams
from tripmaster.core.system.system import is_multi_system

from tripmaster import logging

logger = logging.getLogger(__name__)

class TMReinforceApp(TMConfigurable):
    """
    TMReinforceApp: reinforcement learning application
    """
    EnvironmentPoolType = None
    OutputStreamType = None
    SystemType = None

    def __init__(self, hyper_params, callbacks=None):
        super().__init__(hyper_params)

        self.callbacks = callbacks

        if self.callbacks is None:
            self.callbacks = [TMDefaultSystemRuntimeCallback(self.hyper_params)]

        self.env_pool = self.EnvironmentPoolType(self.hyper_params.io.env)
        # self.output_stream = self.OutputStreamType(self.hyper_params.io.output)

        if self.OutputStreamType:
            self.output_data_stream = self.OutputStreamType(self.hyper_params.io.output)

        if is_multi_system(self.SystemType):
            system_hyper_param = self.hyper_params.multisystem
        else:
            system_hyper_param = self.hyper_params.system
        self.system = self.SystemType.create(system_hyper_param,
                                             callbacks=self.callbacks)

        # TODO: review the design of the following code
        # maybe we should let user control the behavior of the system
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

    def run(self):

        runtime_options = self.hyper_params.job

        result = self.system.run(self.env_pool, runtime_options)

        if not self.system.is_learning() and self.output_data_stream is not None:
            self.output_data_stream.write(result)