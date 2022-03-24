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

Note: The function expected_svf_from_policy and irl is referenced from https://github.com/qzed/irl-maxent/blob/master/src/maxent.py (Author: Maximilian Luz).
License: https://github.com/qzed/irl-maxent/blob/master/LICENSE 
"""
import itertools
from pyexpat import features
from typing import cast, Dict, Iterable, Sequence, Tuple

import numpy as np
from pandas import array
import phirl.api as ph
from phirl._comp import Protocol
import irl_maxent as me


def get_n_states(n_actions: int) -> int:
    """Calculates the number of states."""
    return 2**n_actions


State = Tuple[int, ...]
Space = Tuple[State, ...]

class State_space:
    def __init__(self, n_action):
        self.n_actions = n_action
        
    def get_state_space(self) -> Space:
        """Returns the list of states.
        Note:
            The size of the space of states grows as
            `2 ** n_actions`.
        """
        return tuple(cast(State, state) for state in itertools.product([0, 1], repeat=self.n_actions))
    
    def get_action_space(self) -> Space:
        action_space = []
        for i in range(self.n_actions):
            action = [0]*self.n_actions
            action[i] = 1
            action_space.append(tuple(action))

        return action_space

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

    def __init__(self, n_actions: int, SP: State_space) -> None:
        """The constructor method.
        Args:
            n_actions: number of actions in the MDP
        """
        self.n_actions = n_actions
        self.n_states = get_n_states(n_actions)
        SP = State_space(self.n_actions)
        self.state_space = SP.get_state_space()

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
        idx = self.space.index(tuple(state))
        feature = np.zeros(self.n_states, dtype=float)
        feature[idx] = 1
        return feature

class StateTransitions:

    def __init__(self, n_actions: int, trees: dict, SP: State_space) -> None:
        """The constructor method.
        Args:
            n_actions: number of actions
            trees: a dictionary mapping the tree's ID to the root node.
        """
        self.n_actions = n_actions
        self.n_states = get_n_states(n_actions)
        SP = State_space(n_actions)
        self.state_space = SP.get_state_space()
        self.trees = trees
    
    def get_trajectories(self) -> list:
        """
        returns:
            a list of trajectories that includes nodes and mutations from root to leaf
        """
        trajectories = []
        for tree_id, tree_node in self.trees.items():
            trajectory = ph.list_all_trajectories(tree_node, max_length=20)
            trajectories.append(trajectory)

        return trajectories

    def get_transition(self) -> list:
        """
        returns:
            a 2D list that records the order of states in each trajectory
            (e.g. all_transition[0][0] --> the initial state of the first trajectory)
        """
        trajectories = self.get_trajectories()
        transitions = []
        for trajectory in trajectories: 
            single_transition = [[0, 0, 0, 0, 0]]
            state = [0, 0, 0, 0, 0]
        
            for j in range(len(trajectory[0])):
                if trajectory[0][j].mutation > 0:
                    state[trajectory[0][j].mutation - 1] = 1
                    single_transition.append(state[:])
            transitions.append(single_transition)
        return transitions

    def get_p_transition(self) -> np.array:
        """
        returns:
            p_transition: `[from: Integer, to: Integer, action: Integer] -> probability: Float`
            The probability of a transition from state `from` to state `to` via action `action` to succeed.

            p_action: `[state: Integer, action: Integer] -> probability: Float`
            Local action probabilities
        """
        transitions = self.get_transition()
        p_transition = np.zeros((self.n_states, self.n_states, self.n_actions))

        for i in range(len(transitions)):
            for j in range(len(transitions[i]) - 1):
                current = self.state_space.index(tuple(transitions[i][j]))
                next = self.state_space.index(tuple(transitions[i][j + 1]))
                for n in range(self.n_actions):
                    if transitions[i][j - 1][n] != transitions[i][j][n]:
                        action = n
                        p_transition[current, next, action] += 1

        for i in range(self.n_states):
            if sum(sum(p_transition[i, :, :])) != 0:
                p_transition[i, :, :] /= sum(sum(p_transition[i, :, :]))

        return p_transition

    def get_p_action(self, reward) -> np.array:
        """
        Compute the local action probabilities (policy) required for the edge
        frequency calculation for maximum entropy reinfocement learning.
        
        Args:
            reward: The reward signal per state as table
                `[state: Integer] -> reward: Float`.
        Returns:
            The local action probabilities (policy) as map
            `[state: Integer, action: Integer] -> probability: Float`
        """
        p_transition = self.get_p_transition()
        #er = np.exp(reward)
        p_action = np.zeros((self.n_states,self.n_actions))

        for i in range(self.n_states):
            for j in range(self.n_actions):
                p_action[i,j] = sum(p_transition[i,:,j]*np.exp(reward[i]))
        
        p_action /= p_action.sum(axis=1)[:,None]
        p_action = np.nan_to_num(p_action, nan = 0)

        return p_action

class Action_transition:

    def __init__(self, n_actions: int, trees: dict, SP: State_space, ST: StateTransitions) -> None:
        """The constructor method.
        Args:
            n_actions: number of actions
            trees: a dictionary mapping the tree's ID to the root node.
        """
        self.n_actions = n_actions
        self.n_states = get_n_states(n_actions)
        SP = State_space(n_actions)
        self.action_space = SP.get_action_space()
        ST = StateTransitions(n_actions, trees, SP)
        self.trajectories = ST.get_trajectories()
        self.state_transition = ST.get_transition()
    
    def get_action_transition(self):
        """
        returns:
            a 2D list that records the order of mutations in each trajectory
            (e.g. action_transition[0][0] --> the first action/mutation of the first trajectory)
        """

        action_transition = []
        for path in self.state_transition:
            actions = []
            for i in range(1,len(path)):
                action = [0]*self.n_actions
                for j in range(self.n_actions):
                    if path[i] != path[i-1]:
                        action[i] = 1
                        actions.append(action)
                action_transition.append(actions)

        return action_transition

    def get_p_transition(self):
        transitions = self.get_action_transition()
        p_transition = np.zeros((self.n_actions, self.n_actions, self.n_actions))

        for i in range(len(transitions)):
            for j in range(len(transitions[i]) - 1):
                current = self.action_space.index(tuple(transitions[i][j]))
                next = self.action_space.index(tuple(transitions[i][j + 1]))
                for n in range(self.n_actions):
                    if transitions[i][j - 1][n] != transitions[i][j][n]:
                        action = n
                        p_transition[current, next, action] += 1

        for i in range(self.n_actions):
            if sum(sum(p_transition[i, :, :])) != 0:
                p_transition[i, :, :] /= sum(sum(p_transition[i, :, :]))

        return p_transition

    def get_p_action(self, reward):
        p_transition = self.get_p_transition()
        er = np.exp(reward)
        p_action = np.zeros((self.n_actions,self.n_actions))

        for i in range(self.n_actions):
            for j in range(self.n_actions):
                p_action[i,j] = sum(p_transition[i,:,j]*er[i])
        
        p_action /= p_action.sum(axis=1)[:,None]
        p_action = np.nan_to_num(p_action, nan = 0)
        return p_action


def get_features(featurizer: Featurizer, state_space):
    """
    Args:
        featurizer: mapping used to get the features for every single state
        state_space: a list of states.
    returns:
        A 2-D np array (n_state x dim_feature) that maps the state to its corresponding feature
    """
    features = []
    for state in state_space:
        features.append(featurizer.transform(state))
    
    return np.array(features)


def expected_empirical_feature_counts_from_trajectories(
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
    #trajectories = list(trajectories)

    # The number of trajectories
    m = len(trajectories)
    #state_space = get_state_space(n_actions=5)
    # Get the default feature vector basing
    # on the first state in the first trajectory
    some_state = trajectories[0][0]

    def get_default_feature() -> np.ndarray:
        """Zeros vector, which will be returned
        for the states that have not been visited at all."""
        return np.zeros_like(featurizer.transform(some_state))

    # Initialize the dictionary with zero vectors
    counts = {state: get_default_feature() for state in mdp.state_space}
    #features = []
    features = np.zeros(len(featurizer.transform(trajectories[0][0])))
    for trajectory in trajectories:
        for state in trajectory:
            feature = featurizer.transform(state)
            counts[tuple(state)] += feature / m
            #features.append(feature / m)
            features += feature/m

    return counts, np.array(features)

class Action_features:
    def __init__(self, action_transitions, n_actions) -> None:
        self.action_transitions = action_transitions
        self.n_actions = n_actions

    def get_action_feature_expectation(self):
        feature_expectation = np.zeros(self.n_actions)
        total_action = 0
        for transition in self.action_transitions:
            for action in transition:
                total_action += 1
                feature_expectation += np.array(action)

        return feature_expectation/total_action

    def get_action_initial_features(self):
        initial_feature = np.zeros(self.n_actions)
        for transition in self.action_transitions:
            initial_feature += np.array(transition[0])
        
        return initial_feature/len(self.action_transitions)


def expected_svf_from_policy(p_transition, p_action, p_initial, eps) -> np.array:
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

    Please note: this function is partially referenced from https://github.com/qzed/irl-maxent/blob/master/src/maxent.py line 63-114.
    """
    n_states, _, n_actions = p_transition.shape
    
    p_transition = [np.array(p_transition[:, :, a]) for a in range(n_actions)]
    # actual forward-computation of state expectations
    d = np.zeros(n_states)

    delta = np.inf
    while delta > eps:
        for i in range(n_actions):
            d_ = [p_transition[a].T.dot(p_action[:, a] * d) for a in range(n_actions)]
            d_ = p_initial + np.array(d_).sum(axis=0)
        delta, d = np.max(np.abs(d_ - d)), d_
    return d



def irl(features, feature_expectation, optim, Transition, p_initial, eps, eps_esvf) -> np.array:
    """
    Compute the reward signal given the demonstration trajectories using the
    maximum entropy inverse reinforcement learning algorithm proposed in the
    corresponding paper by Ziebart et al. (2008).

    Args:
        features: The feature-matrix (as a 2D- numpy array), mapping states
        to features, i.e. a matrix of shape (n_states x dim_features).

        feature_expectation: The feature-expectation of the provided trajectories as map
        `[state: Integer] -> feature_expectation: Float`.

        optim: The `Optimizer` instance to use for gradient-based optimization.

        TS: class StateTransition from maxent.py, used to calculate p_action

        eps: The threshold to be used as convergence criterion for the
        reward parameters.

        eps_svf: The threshold to be used as convergence criterion for the
        expected state-visitation frequency.
    
    Returns:
        The reward per state as table `[state: Integer] -> reward: Float`.

    Please note: this function is partially referenced from https://github.com/qzed/irl-maxent/blob/master/src/maxent.py line 196-255.
    """
    p_transition = Transition.get_p_transition()
    theta = np.zeros((len(features[0]),)) + 0.5
    delta = np.inf

    optim.reset(theta)
    #max_iteration = 10000
    #count = 0
    while delta > eps:
        theta_old = theta.copy()
        # compute per-state reward
        reward = features.dot(theta)
        p_action = Transition.get_p_action(reward)

        # compute the gradient
        e_svf = expected_svf_from_policy(p_transition, p_action, p_initial, eps_esvf)
        grad = feature_expectation - features.T.dot(e_svf)

        # perform optimization step and compute delta for convergence
        optim.step(grad)
        delta = np.max(np.abs(theta_old - theta))
        print(theta)
        #count += 1

    #print('D')
    #print(e_svf)
    
    return features.dot(theta)
