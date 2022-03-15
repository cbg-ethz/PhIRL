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
import phirl.api as ph
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

class StateTransitions():

    def __init__(self, n_actions: int, trees: dict) -> None:
        """The constructor method.
        Args:
            n_actions: number of actions
            trees: a dictionary mapping the tree's ID to the root node.
        """
        self.n_actions = n_actions
        self.n_states = get_n_states(n_actions)
        self.state_space = get_state_space(n_actions)
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

    def get_p_action_and_transition(self) -> np.array:
        """
        returns:
            p_transition: `[from: Integer, to: Integer, action: Integer] -> probability: Float`
            The probability of a transition from state `from` to state `to` via action `action` to succeed.

            p_action: `[state: Integer, action: Integer] -> probability: Float`
            Local action probabilities
        """
        transitions = self.get_transition()
        p_transition = np.zeros((self.n_states, self.n_states, self.n_actions))
        p_action = np.zeros((self.n_states, self.n_actions))

        for i in range(len(transitions)):
            for j in range(len(transitions[i]) - 1):
                current = self.state_space.index(tuple(transitions[i][j]))
                next = self.state_space.index(tuple(transitions[i][j + 1]))
                for n in range(self.n_actions):
                    if transitions[i][j + 1][n] != transitions[i][j][n]:
                        mutation = n
                        p_transition[current, next, mutation] += 1
                        p_action[current, mutation] += 1

        for i in range(self.n_states):
            if sum(sum(p_transition[i, :, :])) != 0:
                p_transition[i, :, :] /= sum(sum(p_transition[i, :, :]))

            if sum(p_action[i, :]) != 0:
                p_action[i, :] /= sum(p_action[i, :])

        return p_transition, p_action

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
    state_space = get_state_space(n_actions=5)
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
            counts[tuple(state)] += feature / m

    return counts

#def feature_expectation_from_trajectories(transitions, state_space,featurizer):
#    feature_expectation = {state: 0 for state in state_space}
#    m = len(transitions)
#    for trajectory in transitions:
#        for state in trajectory:
#            idx = state_space.index(tuple(state))
#            feature = featurizer.transform(tuple(state))
#            feature_expectation[idx] += feature / m

#    return feature_expectation

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



