from tripmaster.core.system.supervise import TMSuperviseSystem
from tripmaster.core.system.system import TMMultiSystem



def test_multi():

    class TestMultiSystem(TMSuperviseSystem, TMMultiSystemMixin):
        pass

    TestMultiSystem.init_class(TestMultiSystem)