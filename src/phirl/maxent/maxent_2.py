import dataclasses
import itertools
import json
import pathlib
from typing import List, Callable, cast, Dict, Iterable, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from irl_maxent.optimizer import Optimizer

import phirl.anytree_utils as autil
from phirl._comp import Protocol
import phirl.maxent as me


def get_n_states(n_actions: int) -> int:
    """Calculates the number of states."""
    return 2 ** (n_actions - 1) + 1


State = Tuple[int, ...]
Space = Tuple[State, ...]


def get_state_space(n_actions: int) -> Space:
    """Returns the list of states.

    Note:
        The size of the space of states grows as
        `2 ** n_actions`.
    """
    state_space = tuple(
        cast(State, state) for state in itertools.product([0, 1], repeat=n_actions - 1)
    )
    state_space += tuple(["END"])
    return state_space


def initial_state(n_actions: int) -> State:
    """Returns the initial state (no mutations at all)."""
    return cast(State, tuple(0 for _ in range(n_actions - 1)))


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

        if action < self.n_actions:
            new_state = list(state)
            new_state[action - 1] = 1

            return tuple(new_state)

        elif action == self.n_actions:
            new_state = "END"
            return new_state

        elif action <= 0 or action > self.n_actions:
            raise ValueError(f"Actions {action} are from the set 1, ..., {self.n_actions}.")


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


def unroll_trajectories(
    action_trajectories: Iterable[Sequence[Action]],
    n_actions: int,
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
        unroll_trajectory(
            actions=actions, n_actions=n_actions, initial_state=initial_state, mdp=mdp
        )
        for actions in action_trajectories
    ]


def get_state_transition_trajectories(trajectories: Iterable[Sequence[State]]):
    """This function extract the state-only trajectories from `unroll_trajectories` function."""
    state_trajectories = []
    for trajectory in trajectories:
        state_trajectory = trajectory.states + tuple(["END"])
        state_trajectories.append(state_trajectory)

    return state_trajectories


def get_p_transition(n_actions: int, state_space: Space, mdp: DeterministicTreeMDP) -> np.ndarray:
    """

    returns:
        p_transition: `[from: Integer, to: Integer, action: Integer] -> probability: Float`
        The probability of a transition from state `from` to state `to`
        via action `action` to succeed.

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
    n_states: int, n_actions: int, reward: np.ndarray, state_space: Space, iterations: int = 20
) -> np.ndarray:
    """
    Compute the local action probabilities (policy) required for the edge
        frequency calculation for MaxEnt-IRL.

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
        if state != "END":
            for j in range(n_actions):
                if j < n_actions - 1 and state[j] == 0:
                    next_state = list(state)
                    next_state[j] = 1
                    next_state_idx[i, j] = state_space.index(tuple(next_state))

                elif j == n_actions - 1:
                    next_state_idx[i, j] = n_states - 1

    er = np.exp(reward)
    for i in range(iterations):
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


def expected_svf_from_policy(
    n_actions: int, p_transition: np.ndarray, p_action: np.ndarray
) -> np.ndarray:
    n_states = get_n_states(n_actions)
    d = np.zeros((n_states, n_actions))
    d[0, 0] = 1

    # iterate for N steps
    for t in range(1, n_actions):  # longest trajectory: n_action

        # for all states
        for s_to in range(n_states):

            # sum over nonterminal state-action pairs
            for s_from in range(n_states - 1):
                for a in range(n_actions):
                    d[s_to, t] += (
                        d[s_from, t - 1] * p_action[s_from, a] * p_transition[s_from, s_to, a]
                    )

    return d.sum(axis=1)


@dataclasses.dataclass
class HistoryEntry:
    """A single entry in the optimization history.

    Attrs:
        theta: final reward weights, shape (features_dim,)
        grad: gradient of the likelihood with respect to theta. Shape (features_dim,)
        expected_svf: expected state visitation frequencies,
            calculated using the optimal policy for the reward

    Note:
        Changes here should be reflected in `save_output`.
    """

    theta: np.ndarray
    grad: np.ndarray
    expected_svf: np.ndarray


@dataclasses.dataclass
class IRLOutput:
    """MaxEnt-IRL run result.

    Attrs:
        theta: optimal reward weights
        state_rewards: the reward for each state
        history: optimization history

    Note:
        Changes here should be reflected in `save_output`.
    """

    theta: np.ndarray
    state_rewards: np.ndarray
    history: List[HistoryEntry]


def save_output(output: IRLOutput, path: Union[str, pathlib.Path]) -> None:
    """Saves optimization results to the disk.

    Args:
        output: optimization results
        path: directory name where the results should be dumped
    """
    path = pathlib.Path(path)
    path.mkdir(exist_ok=True)

    with open(path / "final_params.json", "w") as f:
        json.dump(
            fp=f,
            obj={
                "theta": output.theta.tolist(),
                "state_rewards": output.state_rewards.tolist(),
            },
        )

    def save_attr(mapping: Callable[[HistoryEntry], np.ndarray], filename: str) -> None:
        file_path = path / filename
        results = np.asarray([mapping(entry) for entry in output.history])
        pd.DataFrame(results).to_csv(file_path, index=False, header=False)

    save_attr(lambda x: x.theta, "history-theta.csv")
    save_attr(lambda x: x.grad, "history-grad.csv")
    save_attr(lambda x: x.expected_svf, "history-expected_svf.csv")


def irl(
    n_actions: int,
    features: np.ndarray,
    feature_expectation: np.ndarray,
    optim: Optimizer,
    mdp: DeterministicTreeMDP,
    eps: float,
    max_iter: int,
) -> IRLOutput:
    """
    Compute the reward signal given the demonstration trajectories using the
    maximum entropy inverse reinforcement learning algorithm proposed in the
    corresponding paper by Ziebart et al. (2008).

    Args:
        features: feature vector for each state, hape (n_states, features_dim)
        feature_expectation: The feature-expectation of the provided trajectories as map
            `[state: Integer] -> feature_expectation: Float`.
        optim: The `Optimizer` instance to use for gradient-based optimization.
        eps: The threshold to be used as convergence criterion for the
            reward parameters.


    Returns:
        The reward per state as table `[state: Integer] -> reward: Float`.
    """
    n_states = get_n_states(n_actions)
    state_space = get_state_space(n_actions)
    p_transition = get_p_transition(n_actions=n_actions, state_space=state_space, mdp=mdp)

    theta = np.zeros((len(features[0]),)) + 0.5
    optim.reset(theta)

    history: List[HistoryEntry] = []

    # Optimization convergence variables
    delta = np.inf  # Difference between thetas, used to check the convergence
    iteration = 0  # Iteration counter

    while delta > eps and iteration < max_iter:
        iteration += 1

        theta_old = theta.copy()

        # Compute the reward for each state
        reward = features.dot(theta)

        # Get P(a | s), i.e., the policy from the reward
        p_action = get_p_action(n_states, n_actions, reward=reward, state_space=state_space)

        # Compute the likelihood gradient, i.e., eq. 6 in Ziebart (2008)
        e_svf = expected_svf_from_policy(
            n_actions=n_actions, p_transition=p_transition, p_action=p_action
        )
        grad = feature_expectation - features.T.dot(e_svf)

        # Perform optimization step
        optim.step(grad)

        # Check for convergence
        delta = np.linalg.norm(theta_old - theta)

        # Add entry to the history
        history.append(
            HistoryEntry(theta=theta.copy(), grad=grad.copy(), expected_svf=e_svf.copy())
        )

    return IRLOutput(
        theta=theta,
        state_rewards=features.dot(theta),
        history=history,
    )


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
