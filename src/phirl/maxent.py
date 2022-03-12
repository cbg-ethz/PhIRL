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
from typing import cast, Dict, Iterable, Sequence, Tuple

import numpy as np

from phirl._comp import Protocol


def get_n_states(n_actions: int) -> int:
    """Calculates the number of states."""
    return 2**n_actions


State = Tuple[int, ...]
Space = Tuple[State, ...]


def get_state_space(n_actions: int) -> Space:
    """Returns the list of states.

    Note:
        The size of the space of states grows as
        `2 ** n_actions`.
    """
    return tuple(cast(State, state) for state in itertools.product([0, 1], repeat=n_actions))


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


class Featurizer(Protocol):
    """Interface for class with a method
    changing state into the feature vector.

    Every subclass should implement
    `transform` method.
    """

    def transform(self, state: State) -> np.ndarray:
        pass


class IdentityFeaturizer(Featurizer):
    """Feature vector is just the state vector,
    modulo type change.

    The feature space dimension is the number of actions.
    """

    def transform(self, state: State) -> np.ndarray:
        return np.asarray(state, dtype=float)


class OneHotFeaturizer(Featurizer):
    """Each state is encoded using one-hot
    encoding.

    The feature space dimension is 2 raised to the power
    of the number of actions.
    """

    def __init__(self, space: Space) -> None:
        self.space = space
        self.n_states = len(space)

    def transform(self, state: State) -> np.ndarray:
        idx = self.space.index(state)
        feature = np.zeros(self.n_states, dtype=float)
        feature[idx] = 1
        return feature


def expected_empirical_feature_counts(
    mdp: DeterministicTreeMDP,
    featurizer: Featurizer,
    trajectories: Iterable[
        Sequence[State],
    ],
) -> Dict[State, np.ndarray]:
    """Counts expected empirical feature counts ("f tilde").

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

    # Get the default feature vector basing
    # on the first state in the first trajectory
    some_state = trajectories[0][0]

    def get_default_feature() -> np.ndarray:
        """Zeros vector, which will be returned
        for the states that have not been visited at all."""
        return np.zeros_like(featurizer.transform(some_state))

    # Initialize the dictionary with zero vectors
    counts = {state: get_default_feature() for state in mdp.state_space}

    for trajectory in trajectories:
        for state in trajectory:
            feature = featurizer.transform(state)
            counts[state] += feature / m

    return counts
