"""
gpu cloud job
"""
import copy
import os
import addict

from tripmaster import logging
from omegaconf import OmegaConf, DictConfig

from tripmaster.core.concepts.hyper_params import TMHyperParams
from tripmaster.core.system.system import TMSystemRuntimeCallbackInterface

logger = logging.getLogger(__name__)

def trial_name_string(trial):
    """
    Args:
        trial (Trial): A generated trial object.

    Returns:
        trial_name (str): String representation of Trial.
    """
    return trial.trial_id


class TMJob(object):
    """
    NeuralJob
    """

    def __init__(self, app_type,
                 hyper_params,
                 startup_script_path, job_name, args):


        self.app_type = app_type

        self.app_type.SystemType.init_class()

        self.hyper_params = hyper_params # self.app_type.hyper_parameters(self.conf_path, args)

        self.startup_script_path = startup_script_path

        self.job_name = job_name

        from datetime import datetime
        now = datetime.now()
        self.created_time_str = now.strftime("%Y-%m-%d-%H-%M-%S")

    @property
    def mode(self):

        return self.hyper_params.job.mode 

    @property
    def startup_path(self):

        return self.hyper_params.job.startup_path

    @property
    def required_gpus(self):

        return self.hyper_params.resource.computing.gpus

    def name(self):
        """
        name
        """
        
        return self.app_type.__name__ + "-" + self.job_name + "-" + self.created_time_str

    def process_conf(self, conf):
        """
        tuning_conf
        """
        d = dict(locals(), **globals())

        tuning_params = set()

        def __process_tune(source_conf, tuning_config, tuning_params_):
            for key, value in list(source_conf.items()):
                if key.endswith(".tune"):
                    original_param = key.rsplit(".")[0]
                    setattr(tuning_config, original_param, eval(f"exec('from ray import tune') or {value}", d, d))
                    tuning_params_.add(original_param)

                elif isinstance(value, (OmegaConf, DictConfig, dict)):
                    __process_tune(value, getattr(tuning_config, key), tuning_params_)
                else:
                    setattr(tuning_config, key, copy.deepcopy(value))

        tuning_config = TMHyperParams()
        __process_tune(conf, tuning_config, tuning_params)

        return tuning_config, tuning_params

    def run(self):
        """

        Args:
            app_conf ():

        Returns:

        """

        app = self.app_type(self.hyper_params)

        job_conf = self.hyper_params.job
        if job_conf.testing:
            app.test(job_conf.test)
            logger.info("App is set to test mode ...")

        if not job_conf.output_path:
            job_conf.output_path = "tripmaster_results"

        if job_conf.ray_tune:

            import ray     
            from ray import tune

            from ray.tune import CLIReporter
            from ray.tune.schedulers import ASHAScheduler
            #os.environ["RAY_USE_MULTIPROCESSING_CPU_COUNT"] = "1"

            ray.init(num_gpus=job_conf.resource.computing.gpus,
                     num_cpus=job_conf.resource.computing.cpus, include_dashboard=False)


            tuning_config, tuning_params = self.process_conf(self.hyper_params)

            class RayTuneReportCallback(TMSystemRuntimeCallbackInterface):
                def on_evaluation_finished(self, system, evaluation_results):

                    result = evaluation_results[system.hyper_params.learner.modelselector.stage]
                    channel_result = result.performance[system.hyper_params.learner.modelselector.channel]
                    performance = channel_result[system.hyper_params.learner.modelselector.metric]

                    ray.tune.report(model_performance=performance)

            app.system.callbacks.append(RayTuneReportCallback(None))

            scheduler = ASHAScheduler(
                max_t=tuning_config.system.learner.optimizer.strategy.epochs,
                grace_period=1,
                reduction_factor=2)

            reporter = CLIReporter(
                parameter_columns=list(tuning_params),
                metric_columns=["model_performance"])

            startup_path = job_conf.startup_path
            analysis = tune.run(
                lambda config: app.run(),
                config=tuning_config,
                fail_fast=True,
                log_to_file=True,
                local_dir=os.path.join(startup_path, os.path.join(job_conf.output_path, self.job_name)),
                name=self.name(),
                trial_name_creator=trial_name_string,
                resources_per_trial={"cpu": job_conf.resource.computing.cpu_per_trial,
                                     "gpu": job_conf.resource.computing.gpu_per_trial},
                metric="model_performance",
                mode=self.hyper_params.system.learner.modelselector.better,
                scheduler=scheduler,
                progress_reporter=reporter,
            )
            
            logger.info("Best config: ", analysis.best_config)
        else:

            startup_path = job_conf.startup_path
 
            working_directory = os.path.join(startup_path, job_conf.output_path, self.job_name,
                                             self.app_type.__name__, self.created_time_str)

            if os.path.exists(working_directory):
                logger.warning(f"the working directory {working_directory} existed, which may cause file over-written")
            else:
                os.makedirs(working_directory)

            # short_cut_path = os.path.join(startup_path, conf.output_path, self.job_name,
            #                                 self.app_type.__name__, "latest")

            # if os.path.exists(short_cut_path):
            #     if os.path.islink(short_cut_path):
            #         os.remove(short_cut_path)
            #     else:
            #         logger.warning(f"Cannot create the short {short_cut_path} to working directory")
            #
            # if not os.path.exists(short_cut_path):
            #     os.symlink(working_directory, short_cut_path)

            os.chdir(working_directory)

            app.run()
