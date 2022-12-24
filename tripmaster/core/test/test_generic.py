import typing
import inspect
from typing import TypeVar, Generic, List, get_args

def generic_getattr(self, attr):
    """
    Allows classmethods to get generic types
    by checking if we are getting a descriptor type
    and if we are, we pass in the generic type as the class
    instead of the origin type.

    Modified from
    https://github.com/python/cpython/blob/aa73841a8fdded4a462d045d1eb03899cbeecd65/Lib/typing.py#L694-L699
    """

    if "__origin__" in self.__dict__ and not typing._is_dunder(attr):  # type: ignore
        # If the attribute is a descriptor, pass in the generic class
        property = self.__origin__.__getattribute__(self.__origin__, attr)
        if hasattr(property, "__get__"):
            return property.__get__(None, self)
        # Otherwise, just resolve it normally
        return getattr(self.__origin__, attr)
    raise AttributeError(attr)

#typing._GenericAlias.__getattr__ = generic_getattr  # type: ignore

from typing import Generic, TypeVar

def test_generic_type_runtime():
    T = TypeVar('T')
    S = TypeVar('S')

    class Base(Generic[T]):

        @classmethod
        def get_generic_type(cls):
            return typing.get_args(cls)

    class Derived(Base[int], Generic[S]):
        pass

    class DerivedDerived(Generic[S]):
        pass

    print("Base[int].__dict__", Base[int].__dict__)
    print("Derived.__dict__", Derived.__dict__)
    print("Derived[float].__dict__", Derived[float].__dict__)
    print("Derived[float].__origin__.__dict__", Derived[float].__origin__.__dict__)
    print("Derived[float].__args__", Derived[float].__args__)
    print("Derived[float].__origin__.__orig_bases__[0].__args__", Derived[float].__origin__.__orig_bases__[0].__args__)
    print("DerivedDerived.__orig_bases__[0].__args__", DerivedDerived.__orig_bases__[0].__args__)
    print("DerivedDerived[Derived[float]].__origin__.__orig_bases__[0].__args__",
          DerivedDerived[Derived[float]].__origin__.__orig_bases__[0].__args__)
    print(DerivedDerived[Derived[float]].__dict__)
    print(DerivedDerived[Derived[float]].__origin__.__orig_bases__[0].__args__)
    print(Derived.get_generic_type())
    print(Derived[float].get_generic_type())

    a = Derived[float]()
    print(a.__orig_class__.__dict__)
