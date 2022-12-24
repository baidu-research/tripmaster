

from tripmaster.core.launcher.launcher import TMJobLauncher, TMLauncherFactory

from tripmaster.plugins.load import load_plugins

register_funcs = {TMJobLauncher: TMLauncherFactory.get().register}

load_plugins(__name__, __file__, register_funcs)
