"""Script running the MaxEnt IRL
algorithm on a trees data set.
Usage:
    python scripts/run_max_ent.py data=$PWD/path_to_trees.csv n_action=5
Note:
    The data path must be absolute not relative.
"""
import dataclasses
import logging
from typing import Any, Dict, Optional

import anytree
import pandas as pd
import numpy as np

import phirl.api as ph
import phirl.hydra_utils as hy
from phirl.maxent import *
from optimizer import *


# data = pd.read_csv("tree_df.csv", delimiter=",")
# state_combinations = [list(i) for i in itertools.product([0, 1], repeat=5)]
# n_states = 32
# n_action = 5


def get_all_trajectory_states_features(trees, n_states, state_space):
    """
        Args:
            trees: is actually a dictionary mapping the tree's ID to the root node.
        
        Returns:
        all_trajectory: a list of trajectories that includes nodes and mutations from root to leaf
        all_states: a list of all terminal states for each trajectory
        all_features: a list that records whether a state (including transit state) exists in a trajectory
        all_transition: a 2D list that records the order of states in each trajectory
        (e.g. all_transition[0][0] --> the initial state of the first trajectory)
    """
    all_trajectory = []
    all_states = []
    all_features = []
    all_transitions = []

    for tree_id, tree_node in trees.items():
        # trajectory
        trajectory = ph.list_all_trajectories(tree_node, max_length=20)
        all_trajectory.append(trajectory)

        # state and features
        state = [0, 0, 0, 0, 0]
        features = [0] * n_states
        # initial state
        features[0] = 1
        transition = [[0, 0, 0, 0, 0]]

        for j in range(len(trajectory[0])):
            if trajectory[0][j].mutation > 0:
                state[trajectory[0][j].mutation - 1] = 1
                transition.append(state[:])
                idx = state_space.index(tuple(state))
                features[idx] = 1

        all_transitions.append(transition)
        all_states.append(state)
        all_features.append(features)

    return all_trajectory, all_features, all_states, all_transitions


def transition_probability(all_transitions, n_states, n_action, state_space):
    """
        Args: 
            all_transitions: a 2D list that records the order of states in each trajectory

        returns:
            p_transition: `[from: Integer, to: Integer, action: Integer] -> probability: Float`
            The probability of a transition from state `from` to state `to` via action `action` to succeed.

            p_action: `[state: Integer, action: Integer] -> probability: Float`
            Local action probabilities
    """

    p_transition = np.zeros((n_states, n_states, n_action))
    p_action = np.zeros((n_states, n_action))

    for i in range(len(all_transitions)):
        for j in range(len(all_transitions[i]) - 1):
            current = state_space.index(tuple(all_transitions[i][j]))
            next = state_space.index(tuple(all_transitions[i][j + 1]))
            for n in range(n_action):
                if all_transitions[i][j + 1][n] != all_transitions[i][j][n]:
                    mutation = n
                    p_transition[current, next, mutation] += 1
                    p_action[current, mutation] += 1

    for i in range(n_states):
        if sum(sum(p_transition[i, :, :])) != 0:
            p_transition[i, :, :] /= sum(sum(p_transition[i, :, :]))

        if sum(p_action[i, :]) != 0:
            p_action[i, :] /= sum(p_action[i, :])

    return p_transition, p_action


def feature_expectation_from_trajectories(all_features, all_trajectory, state_space):

    """
    Compute the feature expectation of the given trajectories.
    Simply counts the number of visitations to each feature-instance and
    divides them by the number of trajectories.
    Args:
        features: The feature-matrix (e.g. as numpy array), mapping states
            to features, i.e. a matrix of shape (n_states x n_features).
        trajectories: A list or iterator of `Trajectory` instances.
    Returns:
        The feature-expectation of the provided trajectories as map
        `[state: Integer] -> feature_expectation: Float`.
    """

    feature_expectation = [0] * len(state_space)

    for i in range(len(all_features)):
        feature_expectation = [a + b for a, b in zip(feature_expectation, all_features[i])]

    feature_expectation = [number / len(all_trajectory) for number in feature_expectation]

    return feature_expectation


def expected_svf_from_policy(p_transition, p_action, eps=1e-5):
    """
    Compute the expected state visitation frequency using the given local
    action probabilities.
    This is the forward pass of Algorithm 1 of the Maximum Entropy IRL paper
    by Ziebart et al. (2008). Alternatively, it can also be found as
    Algorithm 9.3 in in Ziebart's thesis (2010).
    It has been slightly adapted for convergence, by forcing transition
    probabilities from terminal stats to be zero.
    Args:
        p_transition: The transition probabilities of the MDP as table
            `[from: Integer, to: Integer, action: Integer] -> probability: Float`
            specifying the probability of a transition from state `from` to
            state `to` via action `action` to succeed.
        
        p_action: Local action probabilities as map
            `[state: Integer, action: Integer] -> probability: Float`
            as returned by `local_action_probabilities`.

        eps: The threshold to be used as convergence criterion. Convergence
            is assumed if the expected state visitation frequency changes
            less than the threshold on all states in a single iteration.
    Returns:
        The expected state visitation frequencies as map
        `[state: Integer] -> svf: Float`.
    """
    n_states, _, n_actions = p_transition.shape

    p_initial = np.zeros(n_states)
    p_initial[0] = 1

    p_transition = [np.array(p_transition[:, :, a]) for a in range(5)]
    # actual forward-computation of state expectations
    d = np.zeros(n_states)

    delta = np.inf
    while delta > eps:
        for i in range(n_actions):
            d_ = [p_transition[a].T.dot(p_action[:, a] * d) for a in range(n_actions)]
            d_ = p_initial + np.array(d_).sum(axis=0)
        delta, d = np.max(np.abs(d_ - d)), d_
    return d


# def terminal_state_probabilities(all_transitions):
#    terminals = np.zeros(32)
#    for i in range(len(all_transitions)):
#        idx = state_combinations.index(all_transitions[i][-1])
#        terminals[idx] += 1

#    return terminals/sum(terminals)

''' Not completed
def irl(p_transition, p_action, state_space, all_states, all_features, all_trajectory, optim, eps=1e-4, eps_esvf=1e-5):
    n_states, _, n_actions = p_transition.shape
    p_initial = np.zeros(n_states)
    p_initial[0] = 1

    feature_expectation = feature_expectation_from_trajectories(all_features, all_trajectory, state_space)
    all_states = np.array(all_states)
    all_features = np.array(all_features)
    
    # basic gradient descent
    theta = np.zeros((n_states,)) + 0.5
    delta = np.inf

    optim.reset(theta)
    while delta > eps:
        theta_old = theta.copy()

        # compute per-state reward
        learned_features = np.zeros(n_states)
        reward = learned_features.dot(theta)

        # compute the gradient
        e_svf = expected_svf_from_policy(p_transition, p_action, eps_esvf)

        for i in range(len(all_features)):
            for j in range(n_states):
                learned_features[j] += e_svf[j]*all_features[i][j]
        learned_features /= len(all_features)

        grad = feature_expectation - learned_features

        # perform optimization step and compute delta for convergence
        optim.step(grad)
        delta = np.max(np.abs(theta_old - theta))

    return reward
'''
###
# ************************************************************************


@hy.config
@dataclasses.dataclass
class MainConfig:
    """Class storing arguments for the script.
    Attrs:
        data: path to a data frame with trees
        n_action: number of actions (mutations)
        naming: conventions used to name forest
        max_length: use to limit the maximal length of the trajectory
    """

    data: str = '/Users/apple/Desktop/Lab_rotation1/tree_df.csv'
    n_action: int = 5

    naming: ph.ForestNaming = dataclasses.field(default_factory=ph.ForestNaming)
    max_length: Optional[int] = None


def get_trees(config: MainConfig) -> Dict[Any, anytree.Node]:
    # TODO(Jiayi, Pawel): Missing docstring.
    dataframe = pd.read_csv(config.data)
    trees = ph.parse_forest(dataframe, naming=config.naming)
    return trees


@hy.main
def main(config: MainConfig) -> None:
    logger = logging.getLogger(__name__)
    logger.info("Starting new run...")

    logger.info(f"Creating MDP with {config.n_action} actions...")
    mdp = ph.maxent.DeterministicTreeMDP(n_actions=config.n_action)

    logger.info(f"Reading data from {config.data} file...")
    trees = get_trees(config)
    n_states = get_n_states(config.n_action)
    state_space = get_state_space(config.n_action)

    (
        all_trajectory,
        all_features,
        all_states,
        all_transitions,
    ) = get_all_trajectory_states_features(trees, n_states, state_space)

    logger.info("Calculating feature expectations...")
    feature_expectation = feature_expectation_from_trajectories(all_features, all_trajectory,state_space)

    p_transition, p_action = transition_probability(all_transitions,n_states, config.n_action, state_space)
    d = expected_svf_from_policy(p_transition, p_action, eps=1e-5)
    
    logger.info(f"Feature expectation: {feature_expectation}")
    logger.info(f"Expected SVF: {d}")
    logger.info(f"Local action probabilities: {p_action}")
    #logger.info(f"Reward function: {reward}")


if __name__ == "__main__":
    main()