import addict


class TMHyperParams(addict.Dict):

    def __getstate__(self):
        #return self
        state = self.to_dict()
        isFrozen = (hasattr(self, '__frozen') and
                    object.__getattribute__(self, '__frozen'))
        state['__addict__frozen__'] = isFrozen
        return state

    def __setstate__(self, state):
        shouldFreeze = state.pop('__addict__frozen__', False)
        self.update(state)
        self.freeze(shouldFreeze)