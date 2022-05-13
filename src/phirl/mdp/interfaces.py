import abc
from typing import Generic, Tuple, TypeVar

import numpy as np

State = TypeVar("State")
Action = TypeVar("Action")
# The features are represented with numpy arrays
Feature = np.ndarray


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

    def transform(self, state: State) -> Feature:
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
            This function should be preferred to `len(self.states)`,
             as it is often faster.
        """
        return len(self.states)

    @property
    def n_actions(self) -> int:
        """The number of actions.

        Note:
            This function should be preferred to `len(self.actions)`,
             as it is often faster.
        """
        return len(self.actions)
