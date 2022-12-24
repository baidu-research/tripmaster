from tripmaster.core.concepts.component import TMSerializableComponent


class TMLearningEventTrigger(TMSerializableComponent):
    """
    TMEvaluatorTrigger
    """

    def trigger(self, learner):
        """

        Args:
            learner:

        Returns:

        """
        pass


class TMEpochwiseTrigger(TMLearningEventTrigger):
    """
    TMEpochwiseEvaluatorTrigger
    """

    def __init__(self, hyper_params):
        super().__init__(hyper_params)

        self.last_epoch = 0
        self.interval = self.hyper_params.interval

        assert self.interval > 0

    def trigger(self, learner):
        if learner.epoch >= self.last_epoch + self.interval:
            self.last_epoch = learner.epoch
            return True

    def states(self):
        return {"last_epoch": self.last_epoch, "interval": self.interval}

    def load_states(self, states):
        self.last_epoch = states["last_epoch"]
        self.interval = states["interval"]


class TMStepwiseTrigger(TMLearningEventTrigger):
    """
    TMEpochwiseEvaluatorTrigger
    """

    def __init__(self, hyper_params):
        super().__init__(hyper_params)

        self.last_step = 0
        self.interval = self.hyper_params.interval

        assert self.interval > 0

    def trigger(self, learner):
        if learner.step >= self.last_step + self.interval:
            self.last_step = learner.step
            return True

    def states(self):
        return {"last_step": self.last_step, "interval": self.interval}

    def load_states(self, states):
        self.last_step = states["last_step"]
        self.interval = states["interval"]