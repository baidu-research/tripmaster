from tripmaster.core.launcher.launcher import TMJobLauncher


class TMLocalJobLauncher(TMJobLauncher):
    """
    LocalAppRunner
    """

    Name = "local"

    def run(self, job):
        """

        Args:
            app ():
            args ():

        Returns:

        """

        job.run()
