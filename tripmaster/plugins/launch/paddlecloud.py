import dataclasses
from dataclasses import dataclass, field, fields
from typing import List


from tripmaster.core.launcher.launcher import TMJobLauncher
from tripmaster import logging

import os
import glob

logger = logging.getLogger(__name__)

def dereference(f):
    """

    Args:
        f ():

    Returns:

    """
    if os.path.islink(f):
        # by joining with the dirname we are certain to get the absolute path
        return dereference(os.path.join(os.path.dirname(f), os.readlink(f)))
    else:
        return f


def create_packages_archive(packages, filename):
    """
    Create a tar archive which will contain the files for the packages listed in packages.
    """
    import tarfile
    tar = tarfile.open(filename, "w")

    def add(src, dst):
        """

        Args:
            src ():
            dst ():

        Returns:

        """
        # logger.debug('adding to tar: %s -> %s', src, dst)
        tar.add(src, dst)

    def add_files_for_package(sub_package_path, root_package_path, root_package_name):
        """

        Args:
            sub_package_path ():
            root_package_path ():
            root_package_name ():

        Returns:

        """
        for root, dirs, files in os.walk(sub_package_path):
            if '.svn' in dirs:
                dirs.remove('.svn')
            for f in files:
                if not f.endswith(".pyc") and not f.startswith("."):
                    add(dereference(root + "/" + f), root.replace(root_package_path, root_package_name) + "/" + f)

    for package in packages:
        # Put a submodule's entire package in the archive. This is the
        # magic that usually packages everything you need without
        # having to attach packages/modules explicitly
        if not getattr(package, "__path__", None) and '.' in package.__name__:
            package = __import__(package.__name__.rpartition('.')[0], None, None, 'non_empty')

        n = package.__name__.replace(".", "/")

        if getattr(package, "__path__", None):
            # TODO: (BUG) picking only the first path does not
            # properly deal with namespaced packages in different
            # directories
            p = package.__path__[0]

            if p.endswith('.egg') and os.path.isfile(p):
                raise Exception('egg files not supported!!!')
                # Add the entire egg file
                # p = p[:p.find('.egg') + 4]
                # add(dereference(p), os.path.basename(p))

            else:
                # include __init__ files from parent projects
                root = []
                for parent in package.__name__.split('.')[0:-1]:
                    root.append(parent)
                    module_name = '.'.join(root)
                    directory = '/'.join(root)

                    add(dereference(__import__(module_name, None, None, 'non_empty').__path__[0] + "/__init__.py"),
                        directory + "/__init__.py")

                add_files_for_package(p, p, n)

                # include egg-info directories that are parallel:
                for egg_info_path in glob.glob(p + '*.egg-info'):
                    logger.debug(
                        'Adding package metadata to archive for "%s" found at "%s"',
                        package.__name__,
                        egg_info_path
                    )
                    add_files_for_package(egg_info_path, p, n)

        else:
            f = package.__file__
            if f.endswith("pyc"):
                f = f[:-3] + "py"
            if n.find(".") == -1:
                add(dereference(f), os.path.basename(f))
            else:
                add(dereference(f), n + ".py")
    tar.close()



@dataclass
class PaddleCloudConfig:
    """
    cloud mode configurations
    """
    conf_path: str = ""
    startup_script_path: str = ""
    files: List[str] = field(default_factory=lambda: [])

    fs_name: str = ""
    fs_ugi: str = ""
    output_path: str = ""
    afs_remote: str = ""
    afs_local: str = ""
    image_addr: str = ""
    custom_script: str = ""
    extra_option: str = ""

    env_init_script: str = ""
    python_path: str = "python"
    pip_path: str = "pip"

    package_mode: str = ""
    site_packages_path: str = ""
    ignore_packages: List[str] = field(default_factory=lambda: [])

#    python_home: str = ""

    group_name: str = ""
    job_version: str = ""

    gpu_request: int = 1

    @classmethod
    def parse(cls, hyper_params):

        config = PaddleCloudConfig()
        for field in fields(config):
            if field.name in hyper_params:
                setattr(config, field.name, hyper_params[field.name])

        return config



class PaddleCloudJobLauncher(TMJobLauncher):

    """
    GPUClassRunner
    """

    Name = "paddlecloud"

    def __init__(self, hyper_params):
        super().__init__(PaddleCloudConfig.parse(hyper_params))
        self.submmit_directory = None
        self.job = None

        # self.remote_conf_path = "conf.yaml"

    def write_config(self):
        """

        Args:
            job ():

        Returns:

        """
        # create config.ini file

        with open(os.path.join(self.submmit_directory, "config.ini"), "w") as f:

            f.write(f"fs_name = \"{self.hyper_params.fs_name}\"\n")
            f.write(f"fs_ugi = \"{self.hyper_params.fs_ugi}\"\n")

            f.write(f"output_path = {self.hyper_params.output_path}\n")

            afs_remote = self.hyper_params.afs_remote
            afs_local = self.hyper_params.afs_local
            if afs_remote and afs_local:
                f.write(f"mount_afs = \"true\"\n")
                f.write(f"afs_remote_mount_point = \"{afs_remote}\"\n")
                f.write(f"afs_local_mount_point = \"{afs_local}\"\n")

        # TODO: automatically create the output directory if not exist

    def dump_job_modules(self):
        """

        Args:
            job ():

        Returns:

        """
        # self.job.dump(self.submmit_directory)
        # zip all modules
        packages = self.extra_modules()  # + list(_attached_packages)

        # find the module containing the job
        packages.append(__import__(self.job.__module__, None, None, 'dummy'))

        create_packages_archive(packages, os.path.join(self.submmit_directory, 'packages.tar'))


    def extra_modules(self):
        """
        upload extra modules
        """
        import sys
        module_names = sys.modules.keys()
        root_modules = set([x.split(".")[0] for x in module_names])
        user_developed_modules = []

        import sysconfig
        site_packages_path = sysconfig.get_paths()["purelib"]

        for module_name in root_modules:

            if module_name.startswith("__"):
                continue

            module = sys.modules[module_name]
            if not hasattr(module, "__file__") or module.__file__ is None:
                continue

            path = module.__file__
            if not path.startswith(site_packages_path) and "lib/python" not in path:

                user_developed_modules.append(sys.modules[module_name])

        logger.debug(f"extra modules: {[x.__name__ for x in user_developed_modules]}")

        return user_developed_modules

    def write_afs_command(self):
        """
        write_afs_command
        """
        afs_command = """
$HADOOP_HOME/bin/hadoop fs \
-D fs.default.name={fs_name} \
-D hadoop.job.ugi={fs_ugi} "$@"
""".format(fs_name=self.hyper_params.fs_name, fs_ugi=self.hyper_params.fs_ugi)

        with open(os.path.join(self.submmit_directory, "afs.sh"), "w") as f:
            f.write(afs_command)


    def pack_site_packages(self):
        """
        pack_packages
        """
        import subprocess

        afs_commmand = os.path.join(self.submmit_directory, "afs.sh")

        result = subprocess.run(['sh', afs_commmand, '-stat',
                                 self.hyper_params.site_packages_path],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if result.returncode == 0:
            return

        import sysconfig
        local_site_packages_path = sysconfig.get_paths()["purelib"]

        output_filename = os.path.join(self.submmit_directory,
                                       os.path.basename(self.hyper_params.site_packages_path))
        escaped_packages = set(self.hyper_params.ignore_packages)

        escaped_package_paths = set(os.path.join("site-packages", x) for x in escaped_packages)

        def filter_function(tarinfo):
            name = tarinfo.name.lower()
            if any(name == x or name.startswith(x + "-") for x in escaped_package_paths):
                return None
            else:
                return tarinfo

        import tarfile

        with tarfile.open(output_filename, "w:gz") as tar:
            tar.add(local_site_packages_path, filter=filter_function, arcname="site-packages")

        logger.info("Site-packages of current python has been packaged")

        result = subprocess.run(['sh', afs_commmand, '-put', output_filename,
                                 self.hyper_params.site_packages_path],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if result.returncode == 0:
            logger.info("Successfully upload site-packages into afs")
            subprocess.run(['rm', output_filename],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        else:
            message = f"Error in uploading site-packages: {result.stderr.decode('utf-8')}"
            logger.error(message)
            raise Exception("Faied to Submit the Job")

    def save_requirements(self):
        """

        Returns:

        """

        import subprocess

        result = subprocess.run(['pip', "freeze"],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        requirements = result.stdout.decode("utf-8").split("\n")
        escaped_packages = set(self.hyper_params.ignore_packages)

        needed_requirements = [x for x in requirements
                               if not any(x.lower().startswith(y + '-') for y in escaped_packages)]

        with open("{0}/requirements.txt".format(self.submmit_directory), "w") as f:
            f.write("\n".join(needed_requirements))


    def link_files(self):
        """

        Args:
            job ():
            conf_path ():

        Returns:

        """
        # find the path to out runner.py
        # runner_path = os.path.abspath(remote.__file__)
        # if runner_path.endswith("pyc"):
        #     runner_path = runner_path[:-3] + "py"
        # self.runner_path = runner_path

        files = self.hyper_params.files if self.hyper_params.files else []

        files.extend(self.hyper_params.conf_path)
        files.append(self.job.startup_script_path)

        for f in files:
            if os.path.isabs(f):
                source_path = f
            else:
                source_path = os.path.join(self.job.startup_path, f)
            os.symlink(source_path, os.path.join(self.submmit_directory, os.path.basename(source_path)))

    # def write_app_conf(self):
    #     """

    #     Returns:

    #     """
    #     conf_file_path = os.path.join(self.submmit_directory, self.remote_conf_path)
    #     app_conf = copy.deepcopy(self.job.hyper_params)
    #     app_conf.job.startup_path = ""
    #     OmegaConf.save(config=app_conf.to_dict(), f=conf_file_path)

    def write_job(self):
        """

        Args:
            job ():
            conf_path ():

        Returns:

        """

        assert self.hyper_params.package_mode in {"pack_local", "online_install"}

        if self.hyper_params.package_mode == "pack_local":
            assert self.hyper_params.site_packages_path is not None and self.hyper_params.site_packages_path.strip()
            package_script = """
            sh ./afs.sh -get {remote_site_packages_path} ./site-packages.tar.gz
            tar zxf site-packages.tar.gz 
            export PYTHONPATH=`pwd`/site-packages:$PYTHONPATH
            """.format(remote_site_packages_path=self.hyper_params.site_packages_path)
        else: # self.hyper_params.package_mode == "online_install":
            package_script = "{pip_path} install -r requirements.txt".format(pip_path=self.hyper_params.pip_path)

        remote_conf_paths = [os.path.basename(x) for x in self.hyper_params.conf_path]

        command = "{python_path} $WORK_HOME/{runner_file_name} {extra_option} --conf {conf_file_name} --experiment {experiment_name} launcher.type=local".format(
            python_path=self.hyper_params.python_path,
            runner_file_name=os.path.basename(self.job.startup_script_path),
            conf_file_name=" ".join(remote_conf_paths),
            experiment_name=self.job.job_name,
            extra_option=self.hyper_params.extra_option,
    )

        job_script = """
            set -x 
            {env_init_script}
            {package_script}

            if test -f "packages.tar"; then
                tar xf packages.tar
                export PYTHONPATH=`pwd`:$PYTHONPATH
            fi

            {python_path} --version 

            export LANG="en_US.UTF-8"
            export PYTHONIOENCODING=utf-8
            export WORK_HOME=`pwd`
            {custom_script}
            {command}
            """.format(env_init_script=self.hyper_params.env_init_script,
                       python_path=self.hyper_params.python_path,
                       package_script=package_script,
                       custom_script=self.hyper_params.custom_script,
                       command=command)

        with open(os.path.join(self.submmit_directory, "job.sh"), 'w') as f:
            f.write(job_script)

    def write_submmitter(self):
        """

        Returns:

        """


        job_script = """#!/bin/bash
paddlecloud job  train \
--job-name {job_name} \
--job-conf {config_file_path} \
--group-name {group_name} \
--start-cmd "{start_cmd}" \
--file-dir . \
--job-version {job_version}  \
--image-addr {image_addr} \
--k8s-gpu-cards {k8s_gpu_cards} \
        """.format(job_name=self.job.name(),
                   config_file_path=os.path.join(self.submmit_directory, "config.ini"),
                   group_name=self.hyper_params.group_name,
                   start_cmd="sh job.sh",
                   job_version=self.hyper_params.job_version,
                   k8s_gpu_cards=self.hyper_params.gpu_request,
                   image_addr=self.hyper_params.image_addr
                   )

        # --image-addr {image_addr} \
        with open(os.path.join(self.submmit_directory, "run.sh"), 'w') as f:
            f.write(job_script)


    def submit(self):
        """

        Args:
            job ():
            conf_path ():

        Returns:

        """
        import tempfile

        self.submmit_directory = tempfile.mkdtemp()
        logger.info(f"submit directory = {self.submmit_directory}")

        self.write_afs_command()
        if self.hyper_params.package_mode == "pack_local":
            self.pack_site_packages()
        elif self.hyper_params.package_mode == "online_install":
            self.save_requirements()
        else:
            raise Exception(f"bad package_mode setting, must be [pack_local|online_install]")

        self.dump_job_modules()
        self.link_files()

        self.write_config()
        #        self.write_app_conf()

        self.write_job()
        self.write_submmitter()

        this_dir = os.getcwd()

        os.chdir(self.submmit_directory)

        import subprocess

        result = subprocess.run(['sh', os.path.join(self.submmit_directory, "run.sh")],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        os.chdir(this_dir)

        logger.info(result.stdout.decode("utf-8"))
        if result.returncode:
            message = f"Error in submit job: {result.stderr.decode('utf-8')}"
            logger.error(message)
            raise Exception("Faied to Submit the Job")

            # shutil.rmtree(submmit_directory)

    def run(self, job):
        """

        Args:
            app ():
            args ():

        Returns:

        """

        logger.info("Submmiting application to the server")

        self.job = job
        self.submit()
