from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

import phirl.maxent.interfaces as interfaces


def unroll_trajectory(
    actions: Sequence[interfaces.Action],
    initial_state: interfaces.State,
    dynamics: interfaces.IDeterministicTransitionFunction,
) -> interfaces.Trajectory:
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

    return interfaces.Trajectory(states=states, actions=actions)


def expected_empirical_feature_counts(
    state_space: Tuple[interfaces.State, ...],
    featurizer: interfaces.IFeaturizer,
    trajectories: Iterable[interfaces.Trajectory],
) -> Dict[interfaces.State, interfaces.Feature]:
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
    trajectory: interfaces.Trajectory, start: int = 0, end: int = -1
) -> interfaces.Trajectory:
    """This function creates a subtrajectory from a given trajectory.

    It is used when one wants to ignore some of the shared start states or truncate long trajectories.

    Args:
        trajectory:
        start: the index of the first state to be included into the trajectory
        end: the index of the first state *to not be included* in the trajectory

    Returns:
        subtrajectory
    """
    return interfaces.Trajectory(
        states=trajectory.states[start:end],
        actions=trajectory.states[start:end],
    )
