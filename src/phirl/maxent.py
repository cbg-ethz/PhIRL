"""Maximum Entropy IRL, powered by the irl_maxent module:
https://github.com/qzed/irl-maxent/
"""
from typing import List, Sequence, TypeVar

import numpy as np

import phirl.mdp.interfaces as mdp_int
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
    """

    Returns:
        transition matrix, shape (n_states, n_states, n_actions).
         At index (state_from, state_to, action)
         there is the probability P(state_to | state_from, action)
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
