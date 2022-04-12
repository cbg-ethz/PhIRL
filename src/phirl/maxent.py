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
from platform import node
from statistics import mean, median
from turtle import pen
from typing import List, cast, Dict, Iterable, Sequence, Tuple
#import matplotlib.pyplot as plt

import numpy as np
import numpy.random as rn
import phirl.api as ph
from phirl._comp import Protocol

from irl_maxent.optimizer import Optimizer


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


def get_action_space(n_actions):
    action_space = []
    for i in range(n_actions):
        action = [0] * n_actions
        action[i] = 1
        action_space.append(tuple(action))

    return np.array(action_space)


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
    The feature space dimension is 2 raised to the power
    of the number of actions.
    """

    def __init__(self, space: Space) -> None:
        self.space = space
        self.n_states = len(space)
        # Mapping from index (starting at 0) to the state
        self._index_to_state = dict(enumerate(space))
        # The inverse mapping
        self._state_to_index = dict(zip(self._index_to_state.values(), self._index_to_state.keys()))

    def transform(self, state: State) -> np.ndarray:
        idx = self.space.index(tuple(state))
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


Action = List[int]


def get_action_of_trajectories(trees, max_length=20) -> List[List[Action]]:
    """This function generates a list of actions of each trajectory"""
    action_of_trajectories = []
    for tree_node in trees.values():
        action_each_trajectory = ph.list_all_trajectories(tree_node, max_length=20)
        action_of_trajectories.append(action_each_trajectory)

    actions = []
    for action_each_trajectory in action_of_trajectories:
        action = []
        for nodes in action_each_trajectory:
            for node in nodes:
                if node.mutation > 0:
                    action.append(node.mutation)
        actions.append(action)

    return actions


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


def get_state_transition_trajectories(trajectories: Iterable[Sequence[State]]):
    """This function extract the state-only trajectories from `unroll_trajectories` function."""
    state_trajectories = []
    for trajectory in trajectories:
        state_trajectories.append(trajectory.states)

    return state_trajectories


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


def get_p_transition(n_actions: int, state_space: Space, mdp: DeterministicTreeMDP) -> np.ndarray:
    """

    returns:
        p_transition: `[from: Integer, to: Integer, action: Integer] -> probability: Float`
        The probability of a transition from state `from` to state `to` via action `action` to succeed.

        p_action: `[state: Integer, action: Integer] -> probability: Float`
        Local action probabilities
    """
    n_states = get_n_states(n_actions)
    p_transition = np.zeros((n_states, n_states, n_actions))
    for current_state_idx in range(n_states - 1):
        for action in range(n_actions):
            current_state = state_space[current_state_idx]
            next_state = mdp.new_state(current_state, action + 1)
            next_state_idx = state_space.index(next_state)
            p_transition[current_state_idx, next_state_idx, action] = 1

    return p_transition


def get_p_action(
    n_states: int, n_actions: int, reward: np.ndarray, state_space: Space
) -> np.ndarray:
    """
    Compute the local action probabilities (policy) required for the edge
        frequency calculation for maximum entropy reinfocement learning.

        Args:
            n_states:
            reward: The reward signal per state as table
                `[state: Integer] -> reward: Float`.
        Returns:
            The local action probabilities (policy) as map
            `[state: Integer, action: Integer] -> probability: Float`
    """
    zs = np.ones(n_states)
    zs[0] = 0

    za = np.zeros((n_states, n_actions))
    next_state_idx = np.zeros((n_states, n_actions))

    for i in range(n_states):
        state = state_space[i]
        for j in range(n_actions):
            if state[j] == 0:
                next_state = list(state)
                next_state[j] = 1
                next_state_idx[i, j] = state_space.index(tuple(next_state))

    er = np.exp(reward)
    for i in range(20):
        for j in range(n_states):
            for a in range(n_actions):
                if next_state_idx[j, a] != 0:
                    idx = int(next_state_idx[j, a])
                    za[j, a] = zs[idx]

        za = (za.T * er).T
        zs = za.sum(axis=1)
        zs[-1] = 1

    p_action = za / zs[:, None]
    p_action = np.nan_to_num(p_action, nan=0)

    return p_action


def get_features(featurizer: Featurizer, state_space: Space):
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


"""
def expected_svf_from_policy(
    p_transition: np.ndarray, 
    p_action: np.ndarray, 
    p_initial: np.ndarray, 
    eps: float) -> np.ndarray:
    
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
    
    n_states, _, n_actions = p_transition.shape

    p_transition = np.copy(p_transition)
    # set-up transition matrices for each action
    #p_transition[n_states-1, :, :] = 0.0
    p_transition = [np.array(p_transition[:, :, a]) for a in range(n_actions)]
    

    # actual forward-computation of state expectations
    d = np.zeros(n_states)
    

    delta = np.inf
    delta_test = []
    while delta > eps:
        d_ = [p_transition[a].T.dot(p_action[:, a] * d) for a in range(n_actions)]
        d_ = p_initial + np.array(d_).sum(axis=0)
        delta, d = np.max(np.abs(d_ - d)), d_
        delta_test.append(delta)
        
    return d, delta_test
"""


def expected_svf_from_policy(
    n_actions: int, p_transition: np.ndarray, p_action: np.ndarray
) -> np.ndarray:
    n_states = get_n_states(n_actions)
    d = np.zeros((n_states, n_actions + 1))
    d[0, 0] = 1

    # 5. iterate for N steps
    for t in range(1, n_actions + 1):  # longest trajectory: n_action

        # for all states
        for s_to in range(n_states):

            # sum over nonterminal state-action pairs
            for s_from in range(n_states - 1):
                for a in range(n_actions):
                    d[s_to, t] += (
                        d[s_from, t - 1] * p_action[s_from, a] * p_transition[s_from, s_to, a]
                    )

    return d.sum(axis=1)


def irl(
    n_actions: int,
    features: np.ndarray,
    feature_expectation: np.ndarray,
    optim: Optimizer,
    eps: float,
    mdp: DeterministicTreeMDP,
) -> np.ndarray:

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

        eps: The threshold to be used as convergence criterion for the
        reward parameters.

        trajectories: a set of trajectories. Each trajectory is a sequence of states.

    Returns:
        The reward per state as table `[state: Integer] -> reward: Float`.
    """
    n_states = get_n_states(n_actions)
    state_space = get_state_space(n_actions)
    p_transition = get_p_transition(n_actions, state_space, mdp)

    theta = np.zeros((len(features[0]),)) + 0.5
    # theta = rn.uniform(size=(n_states,))
    delta = np.inf

    optim.reset(theta)
    delta_history = []
    history = [[], [], []]  # [[number_iterations],[mean_reward],[mean_grad]]
    theta_history = []
    iteration = 0

    while delta > eps:
        theta_old = theta.copy()
        # compute per-state reward
        reward = features.dot(theta)

        p_action = get_p_action(n_states, n_actions, reward=reward, state_space=state_space)
        # p_action = get_p_action1(n_actions, reward, p_transition)

        # compute the gradient
        # e_svf, _ = expected_svf_from_policy(p_transition, p_action, p_initial, eps_esvf)
        e_svf = expected_svf_from_policy(n_actions, p_transition, p_action)
        grad = feature_expectation - features.T.dot(e_svf)
        # grad = features.T.dot(e_svf) - feature_expectation
        # print(e_svf)
        # perform optimization step and compute delta for convergence
        optim.step(grad)
        # theta += optim * grad

        delta = np.max(np.abs(theta_old - theta))
        delta_history.append(delta)
        iteration += 1
        history[0].append(iteration)
        history[1].append(np.mean(features.dot(theta)))
        history[2].append(np.linalg.norm(grad))
        theta_history.append(np.mean(theta.copy()))
        # print(theta)

    return features.dot(theta), delta_history, history, theta_history


def get_additive_reward(n_actions: int, learned_reward: np.ndarray) -> np.ndarray:

    """This function calcualtes the reward for states by adding the reward(s) of one-action state(s).

    Args:
        n_actions: the number of actions.
        learned_reward: the reward for states learned by MaxEnt IRL algorithm.

    Returns:
        The additive reward of each state.
    """
    state_space = get_state_space(n_actions)
    n_states = get_n_states(n_actions)
    additive_reward = np.zeros(32)

    for i in range(n_states):
        state_idx = state_space.index(state_space[i])
        state = list(state_space[i])
        if sum(state) > 1:
            mutation = [j for j, e in enumerate(state) if e == 1]
            reward = 0

            for x in mutation:
                action = [0] * n_actions
                action[x] = 1
                reward += learned_reward[state_space.index(tuple(action))]

            additive_reward[state_idx] = reward

        else:
            additive_reward[state_idx] = learned_reward[state_idx]

    return additive_reward


def get_action_reward(n_actions, learned_reward):
    """This function returns the reward for each one-action state."""
    state_space = get_state_space(n_actions)
    action_reward = []
    for i in range(n_actions):
        action = [0] * 5
        action[i] = 1
        action_reward.append(learned_reward[state_space.index(tuple(action))])

    return action_reward

"""
def plot_learning_history(learning_history: list, theta_history: list):
    iteration, mean_reward, grad_norm = learning_history

    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.plot(iteration, mean_reward)
    ax1.set(xlabel="The number of iterations", ylabel="mean reward")

    ax2.plot(iteration, grad_norm)
    ax2.set(xlabel="The number of iterations", ylabel="Grad norm")

    plt.show()
"""

