"""
GPU Cloud runner
"""

import os
from typing import Type
import copy

from tripmaster import logging
from tripmaster.core.concepts.component import TMConfigurable
from tripmaster.core.concepts.hyper_params import TMHyperParams


logger = logging.getLogger(__name__)

script_directory = os.path.dirname(os.path.realpath(__file__))

class TMJobLauncher(TMConfigurable):
    """
    Runner
    """
    Name: str = None

    def run(self, job):
        """

        Args:
            app ():
            args ():

        Returns:

        """
        pass


class TMLauncherFactory(object):

    instance = None

    def __init__(self):

        self.uri_launcher_map = dict()

    def register(self, launcher_class):

        self.uri_launcher_map[launcher_class.Name] = launcher_class

    def has_laucher(self, name):

        return name in self.uri_launcher_map

    def choose(self, name) -> Type[TMJobLauncher]:
        return self.uri_launcher_map[name]

    @classmethod
    def get(cls):
        """

        Returns:

        """
        if cls.instance is None:
            cls.instance = TMLauncherFactory()

        return cls.instance


def launch(app_class):


    """

    Args:
        app ():
        args ():

    Returns:

    """

    import sys

    logging.setup(multi_processing=True)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf-template", dest='conf_template', action='store_true')
    parser.set_defaults(conf_template=False)
    parser.add_argument('--conf', nargs='+', default=[])
    parser.add_argument("--contracts", dest='contracts', action='store_true')
    parser.set_defaults(contracts=False)
    parser.add_argument("--experiment")

    parsed_args, rest_args = parser.parse_known_args(sys.argv)

    if parsed_args.conf_template:
        app_class.generate_conf_template()
        return

    if parsed_args.contracts:
        logger.info("checking contracts")
        app_class.check_system_contracts()
        return

    if not parsed_args.experiment:
        message = "Experiment name not set."
        logger.error(message)
        raise Exception(message)

    if not parsed_args.conf:
        message = "config file not set, please check the --conf option "
        logger.error(message)
        raise Exception(message)

    startup_script_path = rest_args[0]

    conf_path = parsed_args.conf

    hyper_params = app_class.parse_hyper_parameters(conf_path, rest_args[1:])
    from tripmaster.core.components.repo import TMRepoManager
    if hyper_params.repo:
        TMRepoManager.create(hyper_params.repo)

    from tripmaster.core.launcher.job import TMJob

    job = TMJob(app_class, hyper_params, startup_script_path, parsed_args.experiment, rest_args[1:])

    launcher_type = hyper_params.launcher.type
    launcher_params = hyper_params.launcher.strategies[launcher_type]
    if launcher_params:
        launcher_params.conf_path = conf_path

    launcher_class = TMLauncherFactory.get().choose(launcher_type)
    launcher = launcher_class(launcher_params)
    launcher.run(job)