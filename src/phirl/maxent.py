"""Maximum Entropy IRL, powered by the irl_maxent module:
https://github.com/qzed/irl-maxent/
"""
from typing import Iterable, List, Sequence, TypeVar

import irl_maxent.trajectory
import numpy as np

import phirl.mdp.interfaces as mdp_int
import phirl.mdp.common as common
import phirl.enumerate as en

_S = TypeVar("_S")
_A = TypeVar("_A")


def get_features(
    params: mdp_int.IMDPParams[_S, _A], featurizer: mdp_int.IFeaturizer[_S]
) -> np.ndarray:
    assert (
        len(featurizer.shape) == 1
    ), f"Feature vector must be a 1-dim numpy array. Was {featurizer.shape}"

    features = np.zeros((params.n_states, featurizer.shape[0]))

    for i, state in en.Enumerate(params.states).enumerate():
        features[i, :] = featurizer.transform(state)

    return features


def get_p_transition(
    params: mdp_int.IMDPParams[_S, _A], dynamics: mdp_int.IDeterministicTransitionFunction[_S, _A]
) -> np.ndarray:
    """Returns the transition probability matrix.

    Args:
        params: MDP parameters
        dynamics: deterministic transition function

    Returns:
        transition matrix, shape (n_states, n_states, n_actions).
         At index (state_from, state_to, action)
         there is the probability P(state_to | state_from, action)

    Note:
        This matrix contains a lot of redundant information -- we have
         `n_actions * n_states` ones and the rest of the entries
         are zeros. (I.e., only `1/n_states` fraction of entries is non-zero).
    """
    p = np.zeros((params.n_states, params.n_states, params.n_actions))

    map_states = en.Enumerate(params.states)
    map_actions = en.Enumerate(params.actions)

    for i, state_from in map_states.enumerate():
        for a, action in map_actions.enumerate():
            new_state = dynamics.transition(state=state_from, action=action)
            j = map_states.to_index(new_state)
            p[i, j, a] = 1

    return p


def get_terminal(params: mdp_int.IMDPParams, terminal_states: Sequence[_S]) -> List[int]:
    mapping = en.Enumerate(params.states)
    return [mapping.to_index(state) for state in terminal_states]


def convert_trajectory(
    params: mdp_int.IMDPParams[_S, _A], trajectory: common.Trajectory[_S, _A]
) -> irl_maxent.trajectory.Trajectory:
    mapping_states = en.Enumerate(params.states)
    mapping_actions = en.Enumerate(params.actions)

    transitions = []

    for i, action in enumerate(trajectory.actions):
        state_from: _S = trajectory.states[i]
        state_to: _S = trajectory.states[i + 1]

        # Remember to pass from state and action
        # objects to 0-indexed indices.
        # We want tuples (state_from, action, state_to)
        transitions.append(
            (
                mapping_states.to_index(state_from),
                mapping_actions.to_index(action),
                mapping_states.to_index(state_to),
            )
        )
    return irl_maxent.trajectory.Trajectory(transitions)


def get_trajectories(
    params: mdp_int.IMDPParams[_S, _A], trajectories: Iterable[common.Trajectory[_S, _A]]
) -> List[irl_maxent.trajectory.Trajectory]:
    return [convert_trajectory(params=params, trajectory=t) for t in trajectories]
