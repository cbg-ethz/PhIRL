import abc
from typing import Generic, Hashable, Iterable, Sequence, Tuple, TypeVar

_T = TypeVar("_T")


class IEnumerate(abc.ABC, Generic[_T]):
    """Interface used to enumerate objects and quickly
    access the index."""

    @abc.abstractmethod
    def to_index(self, obj: _T) -> int:
        pass

    @abc.abstractmethod
    def to_object(self, index: int) -> _T:
        pass

    @abc.abstractmethod
    def enumerate(self) -> Iterable[Tuple[int, _T]]:
        pass


_H = TypeVar("_H", bound=Hashable)


class _EnumerateHashable(Generic[_H]):
    def __init__(self, objects: Sequence[_H], initial: int = 0) -> None:
        self._to_object = dict(enumerate(objects, initial))
        self._to_index = dict(zip(self._to_object.values(), self._to_object.keys()))

    def to_index(self, obj: _H) -> int:
        return self._to_index[obj]

    def to_object(self, index: int) -> _H:
        return self._to_object[index]

    def enumerate(self) -> Sequence[Tuple[int, _H]]:
        for key, value in self._to_object.items():
            yield (key, value)


class Enumerate(IEnumerate, Generic[_T]):
    """Auxiliary class, mapping objects to indexes."""

    def __init__(self, objects: Sequence[_T], initial: int = 0, hashable: bool = True) -> None:
        if not hashable:
            raise NotImplementedError("The objects must be hashable so far.")
        self._enumerate = _EnumerateHashable(objects, initial=initial)

    def to_object(self, index: int) -> _T:
        return self._enumerate.to_object(index)

    def to_index(self, obj: _T) -> int:
        return self._enumerate.to_index(obj)

    def enumerate(self) -> Iterable[Tuple[int, _T]]:
        return self._enumerate.enumerate()
