"""This is the simplest MDP:
- states are binary tuples, indicating whether an action has taken place (or not)
- actions are numbers from the set {1, ..., n_actions}
- the end states are problematic.
"""
import itertools
from typing import cast, Dict, Tuple

import numpy as np

import phirl.maxent.interfaces as interfaces


State = Tuple[int, ...]  # Each state is a binary tuple (0, 1, 0, ...)
Action = int  # Each action is an integer between 1, ..., n_action


class SimpleTransitionFunction(interfaces.IDeterministicTransitionFunction):
    def __init__(self, n_actions: int) -> None:
        self.n_actions = n_actions

    def transition(self, state: State, action: Action) -> State:
        """As our MDP is deterministic, this function returns a new state.

        Args:
            state: current state
            action: action taken by the agent
        Returns
            new state
        """
        if action <= 0 or action > self.n_actions:
            raise ValueError(f"Actions {action} are from the set 1, ..., {self.n_actions}.")

        new_state = list(state)
        new_state[action - 1] = 1
        return tuple(new_state)


class SimpleParams(interfaces.IMDPParams):
    def __init__(self, n_actions: int) -> None:
        self._n_actions = n_actions

    @property
    def states(self) -> Tuple[State]:
        return tuple(
            cast(State, state) for state in itertools.product([0, 1], repeat=self.n_actions)
        )

    @property
    def actions(self) -> Tuple[Action]:
        return tuple([cast(Action, i) for i in range(1, self.n_actions + 1)])

    @property
    def n_states(self) -> int:
        return 2**self.n_actions

    @property
    def n_actions(self) -> int:
        return self._n_actions

    @property
    def initial_state(self) -> State:
        """Returns the initial state (no mutations at all)."""
        return cast(State, tuple(0 for _ in range(self.n_actions)))


class IdentityFeaturizer(interfaces.IFeaturizer):
    """Feature vector is just the state vector,
    modulo type change.

    The feature space dimension is the number of actions.
    """

    def __init__(self, params: SimpleParams) -> None:
        """
        Args:
            n_actions: number of actions
        """
        self._n = params.n_actions

    def transform(self, state: State) -> np.ndarray:
        if len(state) != self._n:
            raise ValueError(f"State {state} should be of length {self._n}.")
        return np.asarray(state, dtype=float)

    @property
    def shape(self) -> Tuple[int, ...]:
        return (self._n,)


class OneHotFeaturizer(interfaces.IFeaturizer):
    """Each state is encoded using one-hot
    encoding.

    The feature space dimension is the number of states
    (i.e., `2**n_actions`).
    """

    def __init__(self, params: SimpleParams) -> None:
        self.space = params.states
        self.n_states = params.n_states
        assert (
            len(self.space) == self.n_states
        ), f"State space length mismatch: {len(self.space)} != {self.n_states}."

        # Mapping from index (starting at 0) to the state
        self._index_to_state: Dict[int, State] = dict(enumerate(self.space))
        # The inverse mapping
        self._state_to_index: Dict[State, int] = dict(
            zip(self._index_to_state.values(), self._index_to_state.keys())
        )

    def transform(self, state: State) -> np.ndarray:
        idx = self.space.index(tuple(state))
        feature = np.zeros(self.n_states, dtype=float)
        feature[idx] = 1
        return feature

    @property
    def shape(self) -> Tuple[int, ...]:
        return (self.n_states,)

    def state_to_index(self, state: State) -> int:
        """Maps the state to its index, which is the
        only non-zero coordinate in the one-hot encoding (starting at 0).

        Args:
            state: state

        Returns:
            index, starting at 0

        See Also:
            index_to_state, the inverse mapping
        """
        return self._state_to_index[state]

    def index_to_state(self, index: int) -> State:
        """For a given `index` returns a state which
        is represented by a one-hot vector with 1 at
        position `index`.

        Args:
            index: number between 0 and `n_states` - 1

        Returns:
            the state which will be represented by the
            one-hot vector specified by `index`

        See Also:
            state_to_index, the inverse mapping
        """
        return self._index_to_state[index]
