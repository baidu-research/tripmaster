"""
data server holding all data set
"""

import os
from enum import auto
from typing import Optional

from tripmaster import logging
from tripmaster.core.concepts.component import TMConfigurable

from tripmaster.utils.enum import AutoNamedEnum

logger = logging.getLogger(__name__)


class TMRepoCategory(AutoNamedEnum):

    Data = auto()
    Resource = auto()

    TaskData = auto()
    Modeler = auto()
    ProblemData = auto()

    Learner = auto()
    Inferencer = auto()
    Machine = auto()

    System = auto()


class TMComponentRepo(TMConfigurable):
    """
    TMRepo
    """

    def __init__(self, hyper_params, category: TMRepoCategory):

        super().__init__(hyper_params)

        self.server = self.hyper_params.server
        self.local_dir = os.path.expanduser(self.hyper_params.local_dir)

        self.category = category
        self.uri_path_map = dict()

    def download(self, uri):
        """

        Args:
            uri ():

        Returns:

        """

        if not self.uri_path_map:
            remote_data_file_path = os.path.join(self.server, self.category.value, "index.txt")
            logger.info(f'Downloading index File from {remote_data_file_path}')

            import wget
            import tempfile
            temp_file_path = tempfile.NamedTemporaryFile().name

            try:
                wget.download(remote_data_file_path, temp_file_path)
            except:

                logger.error("Failed to download index files ")
                raise RuntimeError(f"Failed to download index files for repo {self.category.value}")

            with open(temp_file_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    this_uri, this_path = line.split("\t", 1)
                    self.uri_path_map[this_uri] = this_path

        if uri not in self.uri_path_map:
            raise Exception(f"uri {uri} not in the repo")

        data_path = os.path.join(self.local_dir, self.category.value)

        if os.path.exists(data_path) and os.path.isfile(data_path):

            logger.error(f"the path {data_path} is a file, not a directory")
            raise Exception("the local data directory is not avaliable")
        elif not os.path.exists(data_path):
            logger.warning(f"the path {data_path} does not exist, making the local data directory")
            os.mkdir(data_path)

        remote_data_url = os.path.join(self.server, self.category.value, self.uri_path_map[uri])
        local_data_dir = os.path.join(data_path, uri)

        import wget
        file_name = wget.filename_from_url(remote_data_url)
        local_data_file_path = os.path.join(local_data_dir, file_name)

        if os.path.exists(local_data_file_path):
            logger.warning(f"The path already exists {local_data_file_path}")
            if os.path.isdir(local_data_file_path):
                raise Exception(f"the path {local_data_file_path} is expected to be a file, not a directory")

        else:
            if not os.path.exists(local_data_dir):
                os.makedirs(local_data_dir)

            remote_data_file_path = os.path.join(self.server, self.category.value, self.uri_path_map[uri])
            logger.info(f'Downloading Data File from {remote_data_file_path} to {local_data_file_path}')
            local_data_file_path = wget.download(remote_data_file_path, local_data_dir)

        return local_data_file_path

    def get(self, uri):
        """

        Args:
            uri:

        Returns:

        """

        return self.download(uri)

class TMRepoManager(TMConfigurable):

    manager: Optional["TMRepoManager"] = None

    def __init__(self, hyper_params):

        super().__init__(hyper_params)

        self.local_dir = os.path.expanduser(self.hyper_params.local_dir)

        if not os.path.exists(self.local_dir):
            os.makedirs(self.local_dir)

        self.data_repo = TMComponentRepo(self.hyper_params, TMRepoCategory.Data)
        self.resource_repo = TMComponentRepo(self.hyper_params, TMRepoCategory.Resource)
        self.taskdata_repo = TMComponentRepo(self.hyper_params, TMRepoCategory.TaskData)
        self.modeler_repo = TMComponentRepo(self.hyper_params, TMRepoCategory.Modeler)
        self.problemdata_repo = TMComponentRepo(self.hyper_params, TMRepoCategory.ProblemData)
        self.machine_repo = TMComponentRepo(self.hyper_params, TMRepoCategory.Machine)
        self.learner_repo = TMComponentRepo(self.hyper_params, TMRepoCategory.Learner)
        self.inferencer_repo = TMComponentRepo(self.hyper_params, TMRepoCategory.Inferencer)
        self.system_repo = TMComponentRepo(self.hyper_params, TMRepoCategory.System)

    @classmethod
    def create(cls, hyper_params):
        """

        Args:
            yaml_file_path ():

        Returns:

        """
        cls.manager = TMRepoManager(hyper_params)

        return cls.manager

    @classmethod
    def get(cls):
        """

        Returns:

        """
        return cls.manager


class TMRepo(TMConfigurable):
    """
    TMRepo
    """
    def __init__(self, hyper_params=None):

        super().__init__(hyper_params)

    def get(self, uri):
        """

        Args:
            uri:

        Returns:

        """

        manager = TMRepoManager.get()
        component, component_uri = uri.split(":", 1)
        component = component.lower()

        if component == "data":
            return manager.data_repo.get(component_uri)
        elif component == "resource":
            return manager.resource_repo.get(component_uri)
        elif component == "taskdata":
            return manager.taskdata_repo.get(component_uri)
        elif component == "modeler":
            return manager.modeler_repo.get(component_uri)
        elif component == "problemdata":
            return manager.problemdata_repo.get(component_uri)
        elif component == "machine":
            return manager.machine_repo.get(component_uri)
        elif component == "learner":
            return manager.learner_repo.get(component_uri)
        elif component == "inferencer":
            return manager.inferencer_repo.get(component_uri)
        elif component == "system":
            return manager.system_repo.get(component_uri)
        else:
            raise Exception(f"unknown component {component}")



