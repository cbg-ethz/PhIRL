"""Submodule implementing the MaxEnt IRL
approach to finding the reward function.

The MDP is deterministic and each state
is a set of already acquired mutations.

We will encode each state as a binary tuple
(0, 1, 0, ..., 0),
and 1 on ith position means that the mutation
i+1 has been acquired.

MaxEnt IRL was proposed in
B.D. Ziebart et al., Maximum Entropy Inverse Reinforcement Learning,
AAAI (2008), https://www.aaai.org/Papers/AAAI/2008/AAAI08-227.pdf
"""
import itertools
from typing import cast, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from phirl._comp import Protocol


def get_n_states(n_actions: int) -> int:
    """Calculates the number of states."""
    return 2**n_actions


Action = int
State = Tuple[Action, ...]
Space = Tuple[State, ...]


def get_state_space(n_actions: int) -> Space:
    """Returns the list of states.

    Note:
        The size of the space of states grows as
        `2 ** n_actions`.
    """
    return tuple(cast(State, state) for state in itertools.product([0, 1], repeat=n_actions))


class Trajectory:
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


class DeterministicTreeMDP:
    """This class defines a deterministic MDP.

    We consider `n` actions, numbered from 1 to `n`.
    Each state is a subset of actions, represented
    as `n`-tuple with binary entries, representing whether
    a particular mutation is

    Note:
        As Python is 0-indexed and we index actions from 1,
        if 1 is the `i`th in the state, it means that the
        action `i+1`th is present.
    """

    def __init__(self, n_actions: int) -> None:
        """The constructor method.

        Args:
            n_actions: number of actions in the MDP
        """
        self.n_actions = n_actions
        self.n_states = get_n_states(n_actions)
        self.state_space = get_state_space(n_actions)

    def new_state(self, state: State, action: int) -> State:
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


def unroll_trajectory(
    actions: Sequence[Action], initial_state: State, mdp: DeterministicTreeMDP
) -> Trajectory:
    """Using a *deterministic* MDP simulates the trajectory, basing on executed `actions`
    starting at `initial_state`.

    Args:
        actions: a sequence of actions
        initial_state: starting state
        mdp: deterministic transition function

    Returns:
        the trajectory (both states and actions)

    See Also:
        unroll_trajectories, a convenient function for generating multiple trajectories
    """
    states = [initial_state]
    for action in actions:
        new_state = mdp.new_state(state=states[-1], action=action)
        states.append(new_state)

    return Trajectory(states=states, actions=actions)


def unroll_trajectories(
    action_trajectories: Iterable[Sequence[Action]],
    initial_state: State,
    mdp: DeterministicTreeMDP,
) -> List[Trajectory]:
    """This function applies `unroll_trajectory` to each action sequence
    in `action_trajectories`, assuming that all of these start at `initial_state`
    and follow a deterministic transition function (`mdp`).

    Note:
        The `initial_state` needs to be immutable, as we don't copy it to each trajectory.

    See Also:
        unroll_trajectory, the backend of this function
    """
    return [
        unroll_trajectory(actions=actions, initial_state=initial_state, mdp=mdp)
        for actions in action_trajectories
    ]


class Featurizer(Protocol):
    """Interface for class with a method
    changing state into the feature vector.

    Every subclass should implement
    `transform` method (mapping state to the feature vector)
    and `shape`, returning the shape of the features.
    """

    def transform(self, state: State) -> np.ndarray:
        raise NotImplementedError

    @property
    def shape(self) -> Tuple[int, ...]:
        raise NotImplementedError


class IdentityFeaturizer(Featurizer):
    """Feature vector is just the state vector,
    modulo type change.

    The feature space dimension is the number of actions.
    """

    def __init__(self, n_actions: int) -> None:
        """

        Args:
            n_actions: number of actions
        """
        self._n = n_actions

    def transform(self, state: State) -> np.ndarray:
        if len(state) != self._n:
            raise ValueError(f"State {state} should be of length {self._n}.")
        return np.asarray(state, dtype=float)

    @property
    def shape(self) -> Tuple[int, ...]:
        return (self._n,)


class OneHotFeaturizer(Featurizer):
    """Each state is encoded using one-hot
    encoding.

    The feature space dimension is the number of states,
    that is 2 raised to the power
    of the number of actions.
    """

    def __init__(self, space: Space) -> None:
        self.n_states = len(space)

        # Mapping from index (starting at 0) to the state
        self._index_to_state = dict(enumerate(space))
        # The inverse mapping
        self._state_to_index = dict(zip(self._index_to_state.values(), self._index_to_state.keys()))

    def transform(self, state: State) -> np.ndarray:
        idx = self.state_to_index(state)
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


def expected_empirical_feature_counts(
    mdp: DeterministicTreeMDP,
    featurizer: Featurizer,
    trajectories: Iterable[
        Sequence[State],
    ],
) -> Dict[State, np.ndarray]:
    """Counts expected empirical feature counts, which is
    `\\tilde f` on page 2 (1434)
    in B.D. Ziebart et al., Maximum Entropy Inverse Reinforcement Learning.

    Args:
        mdp: underlying deterministic MDP
        featurizer: mapping used to get the features for every single state
        trajectories: a set of trajectories. Each trajectory is a sequence of states.

    Returns:
        dictionary mapping all states in MDP to their expected empirical
        feature counts
    """
    trajectories = list(trajectories)

    # The number of trajectories
    m = len(trajectories)

    def get_default_feature() -> np.ndarray:
        """Zeros vector, which will be returned
        for the states that have not been visited at all."""
        return np.zeros(featurizer.shape)

    # Initialize the dictionary with zero vectors
    counts = {state: get_default_feature() for state in mdp.state_space}

    for trajectory in trajectories:
        for state in trajectory:
            feature = featurizer.transform(state)
            counts[state] += feature / m

    return counts
