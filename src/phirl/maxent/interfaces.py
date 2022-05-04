import abc
from typing import Generic, Sequence, Tuple, TypeVar

import numpy as np

State = TypeVar("State")
Action = TypeVar("Action")


class IDeterministicTransitionFunction(abc.ABC, Generic[State, Action]):
    """Interface for a deterministic transition function.

    Methods:
        transition: the deterministic transition function
    """

    def transition(self, state: State, action: Action) -> State:
        """The deterministic transition function.

        Args:
            state: old state
            action: executed action

        Returns:
            the new state (obtained by executing `action` on `state`)
        """
        pass


class IFeaturizer(abc.ABC, Generic[State]):
    """Interface for class with a method mapping
    the state into the feature vector."""

    def transform(self, state: State) -> np.ndarray:
        """Maps the state to the feature vector."""
        raise NotImplementedError

    @property
    def shape(self) -> Tuple[int, ...]:
        """The shape of the returned feature vector."""
        raise NotImplementedError


class IMDPParams(abc.ABC, Generic[State, Action]):
    @property
    @abc.abstractmethod
    def states(self) -> Tuple[State]:
        """Returns a tuple with all states possible."""
        pass

    @property
    @abc.abstractmethod
    def actions(self) -> Tuple[Action]:
        """Returns a tuple with all actions possible."""
        pass

    @property
    def n_states(self) -> int:
        """The number of states.

        Note:
            This function should be preferred to `len(self.states())`,
             as it is often faster.
        """
        return len(self.states)

    @property
    def n_actions(self) -> int:
        """The number of actions.

        Note:
            This function should be preferred to `len(self.actions())`,
             as it is often faster.
        """
        return len(self.actions)


class Trajectory(Generic[State, Action]):
    """An object representing an MDP trajectory.

    Attrs:
        states: a tuple of states visited by agent, length n
        actions: tuple of actions executed by agent, length n-1

    Note:
        1. `actions[k]` corresponds to the action executed by the agent between
          `states[k] and `states[k+1]`
        2. The `states` and `actions` do *not* have equal lengths.
    """

    def __init__(self, states: Sequence[State], actions: Sequence[Action]) -> None:
        self.states: Tuple[State] = tuple(states)
        self.actions: Tuple[Action] = tuple(actions)

        if len(actions) != len(states) - 1:
            raise ValueError("Length of actions must be the length of states minus 1.")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(states={self.states}, actions={self.actions})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, type(self)):
            return False
        return self.states == other.states and self.actions == other.actions
