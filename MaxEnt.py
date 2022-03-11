from matplotlib.pyplot import get
import phirl.api as ph
import pandas as pd
import numpy as np
import itertools

import trajectory

data = pd.read_csv("tree_df.csv", delimiter=",")
mutation = pd.read_csv("mhn.csv", delimiter=",")
state_combinations = [list(i) for i in itertools.product([0, 1], repeat=5)]
n_states = 32
n_action = 5


def get_trees():
    Forest_naming = ph.ForestNaming()
    trees = ph.parse_forest(data, Forest_naming)
    return trees


def get_all_trajectory_states_features(trees):
    all_trajectory = []
    all_states = []
    all_features = []
    all_transitions = []

    for i in range(1, len(trees) + 1):
        # trajectory
        trajectory = ph.list_all_trajectories(trees[i], max_length=10)
        all_trajectory.append(trajectory)

        # state and features
        state = [0, 0, 0, 0, 0]
        features = [0] * n_states
        transition = [[0, 0, 0, 0, 0]]

        for j in range(len(trajectory[0])):
            if trajectory[0][j].mutation > 0:

                state[trajectory[0][j].mutation - 1] = 1
                transition.append(state[:])
                idx = state_combinations.index(state)
                features[idx] = 1

        all_transitions.append(transition)
        all_states.append(state)
        all_features.append(features)

    return all_trajectory, all_features, all_states, all_transitions

def transition_probability(all_transitions):
    p_transition = np.zeros((n_states, n_states, n_action))
    p_action = np.zeros((n_states, n_action))

    for i in range(len(all_transitions)):
        for j in range(len(all_transitions[i]) - 1):
            current = state_combinations.index(all_transitions[i][j])
            next = state_combinations.index(all_transitions[i][j + 1])
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


def feature_expectation_from_trajectories(all_features, all_trajectory):
    feature_expectation = [0] * len(state_combinations)

    for i in range(len(all_features)):
        feature_expectation = [
            a + b for a, b in zip(feature_expectation, all_features[i])
        ]

    feature_expectation = [
        number / len(all_trajectory) for number in feature_expectation
    ]

    return feature_expectation


def expected_svf_from_policy(p_transition, p_action, eps=1e-5):
    p_initial = np.zeros(n_states)
    p_initial[0] = 1

    p_transition = [np.array(p_transition[:, :, a]) for a in range(5)]
    # actual forward-computation of state expectations
    d = np.zeros(n_states)

    delta = np.inf
    while delta > eps:
        for i in range(1, 6):
            d_ = [p_transition[a].T.dot(p_action[:, a] * d) for a in range(5)]
            d_ = p_initial + np.array(d_).sum(axis=0)
        delta, d = np.max(np.abs(d_ - d)), d_
    return d


# def terminal_state_probabilities(all_transitions):
#    terminals = np.zeros(32)
#    for i in range(len(all_transitions)):
#        idx = state_combinations.index(all_transitions[i][-1])
#        terminals[idx] += 1

#    return terminals/sum(terminals)


def local_action_probabilities(p_transition, all_states):

    reward = np.zeros((n_states,)) + 0.5
    er = np.exp(reward)

    p = [np.array(p_transition[:, :, a]) for a in range(n_action)]

    # initialize at terminal states
    zs = np.zeros(n_states)

    for i in range(len(all_states)):
        idx = state_combinations.index(all_states[i])
        zs[idx] = 1

    # perform backward pass
    for _ in range(2 * n_states):
        za = np.array([er * p[a].dot(zs) for a in range(n_action)]).T
        zs = za.sum(axis=1)

    for i in range(len(za)):
        print(za[i])
        
    # print(za / zs[:,None])
    return za / zs[:, None]


if __name__ == "__main__":
    trees = get_trees()
    (
        all_trajectory,
        all_features,
        all_states,
        all_transitions,
    ) = get_all_trajectory_states_features(trees)
    feature_expectation = feature_expectation_from_trajectories(
        all_features, all_trajectory
    )
    p_transition, p_action = transition_probability(all_transitions)
    d = expected_svf_from_policy(p_transition, p_action, eps=1e-5)
    local = local_action_probabilities(p_transition, all_states)
