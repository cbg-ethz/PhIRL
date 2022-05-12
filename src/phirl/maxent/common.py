from typing import Dict, Generic, Hashable, Iterable, Sequence, Tuple, TypeVar

import numpy as np

import phirl.maxent.interfaces as interfaces

_S = TypeVar("_S")  # Placeholder for states
_A = TypeVar("_A")  # Placeholder for actions


class Trajectory(Generic[_S, _A]):
    """An object representing an MDP trajectory.

    Attrs:
        states: a tuple of states visited by agent, length n
        actions: tuple of actions executed by agent, length n-1

    Note:
        1. `actions[k]` corresponds to the action executed by the agent between
          `states[k] and `states[k+1]`
        2. The `states` and `actions` do *not* have equal lengths.
    """

    def __init__(self, states: Sequence[_S], actions: Sequence[_A]) -> None:
        self.states: Tuple[_S] = tuple(states)
        self.actions: Tuple[_A] = tuple(actions)

        if len(actions) != len(states) - 1:
            raise ValueError("Length of actions must be the length of states minus 1.")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(states={self.states}, actions={self.actions})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, type(self)):
            return False
        return self.states == other.states and self.actions == other.actions


def unroll_trajectory(
    actions: Sequence[_A],
    initial_state: _S,
    dynamics: interfaces.IDeterministicTransitionFunction,
) -> Trajectory[_S, _A]:
    """Using a *deterministic* MDP simulates the trajectory, basing on executed `actions`
    starting at `initial_state`.
    Args:
        actions: a sequence of actions
        initial_state: starting state
        dynamics: deterministic transition function

    Returns:
        the trajectory (both states and actions)

    See Also:
        unroll_trajectories, a convenient function for generating multiple trajectories
    """
    states = [initial_state]
    for action in actions:
        new_state = dynamics.transition(state=states[-1], action=action)
        states.append(new_state)

    return Trajectory(states=states, actions=actions)


def expected_empirical_feature_counts(
    state_space: Tuple[_S, ...],
    featurizer: interfaces.IFeaturizer,
    trajectories: Iterable[Trajectory[_S, _A]],
) -> Dict[_S, interfaces.Feature]:
    """Counts expected empirical feature counts, which is
    `\\tilde f` on page 2 (1434)
    in B.D. Ziebart et al., Maximum Entropy Inverse Reinforcement Learning.

    Args:
        state_space: the state space
        featurizer: mapping used to get the features for every single state
        trajectories: a set of trajectories. Each trajectory is a sequence of states.

    Returns:
        dictionary mapping all states in MDP to their expected empirical
        feature counts
    """
    trajectories = list(trajectories)

    # The number of trajectories
    m = len(trajectories)

    def get_default_feature() -> interfaces.Feature:
        """Zeros vector, which will be returned
        for the states that have not been visited at all."""
        return np.zeros(featurizer.shape)

    # Initialize the dictionary with zero vectors
    counts = {state: get_default_feature() for state in state_space}

    for trajectory in trajectories:
        for state in trajectory.states:
            feature = featurizer.transform(state)
            counts[state] += feature / m

    return counts


def slice_trajectory(
    trajectory: Trajectory[_S, _A], start: int = 0, end: int = -1
) -> Trajectory[_S, _A]:
    """This function creates a subtrajectory from a given trajectory.

    It is used when one wants to ignore some of the shared start states
     or truncate long trajectories.

    Args:
        trajectory:
        start: the index of the first state to be included into the trajectory
        end: the index of the first state *to not be included* in the trajectory

    Returns:
        subtrajectory
    """
    return Trajectory(
        states=trajectory.states[start:end],
        actions=trajectory.states[start:end],
    )


_T = TypeVar("_T")
_H = TypeVar("_H", bound=Hashable)


class _EnumerateHashable(Generic[_H]):
    def __init__(self, objects: Sequence[_H], initial: int = 0) -> None:
        self._to_object = dict(enumerate(objects, initial))
        self._to_index = dict(zip(self._to_object.values(), self._to_object.keys()))

    def to_index(self, obj: _H) -> int:
        return self._to_index[obj]

    def to_object(self, index: int) -> _H:
        return self._to_object[index]


class Enumerate(interfaces.IEnumerate, Generic[_T]):
    """Auxiliary class, mapping objects to indexes."""

    def __init__(self, objects: Sequence[_T], initial: int = 0, hashable: bool = True) -> None:
        if not hashable:
            raise NotImplementedError("The objects must be hashable so far.")
        self._enumerate = _EnumerateHashable(objects, initial=initial)

    def to_object(self, index: int) -> _T:
        return self._enumerate.to_object(index)

    def to_index(self, obj: _T) -> int:
        return self._enumerate.to_index(obj)
