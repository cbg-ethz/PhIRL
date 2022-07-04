from random import uniform
import numpy as np
import copy

import phirl.enumerate as en
import phirl.mdp.interfaces as mdp_int
from typing import TypeVar, Iterable
import scipy.stats
import phirl.mdp.common as common

_S = TypeVar("_S")
_A = TypeVar("_A")

def get_p_transition(
    params: mdp_int.IMDPParams[_S, _A], dynamics: mdp_int.IDeterministicTransitionFunction[_S, _A]
) -> np.ndarray:
    """Returns the transition probability matrix.

    Args:
        params: MDP parameters
        dynamics: deterministic transition function

    Returns:
        transition matrix, shape (n_states, n_actions, n_states).
         At index (state_from, state_to, action)
         there is the probability P(state_to | state_from, action)

    Note:
        This matrix contains a lot of redundant information -- we have
         `n_actions * n_states` ones and the rest of the entries
         are zeros. (I.e., only `1/n_states` fraction of entries is non-zero).
    """
    p = np.zeros((params.n_states, params.n_actions, params.n_states))

    map_states = en.Enumerate(params.states)
    map_actions = en.Enumerate(params.actions)

    for i, state_from in map_states.enumerate():
        for a, action in map_actions.enumerate():
            new_state = dynamics.transition(state=state_from, action=action)
            j = map_states.to_index(new_state)
            p[i, a, j] = 1

    return p


def compute_prior(prior, reward, min=-1, max=1, mu=0, sigma=0.1):
    if prior == 'uniform':
        return 1/(max-min)
    elif prior == 'gaussian':
        return np.exp(np.sum(scipy.stats.norm.logpdf(reward,loc=mu, scale = sigma)))
    else:
        raise NotImplementedError('{} is not implemented.'.format(prior))

"""
Please note that the following functions are referenced from:
https://github.com/uidilr/bayesian_irl/blob/master/src/algos/learn.py

Author: Yusuke Nakata
License: https://github.com/uidilr/bayesian_irl/blob/master/LICENSE 
"""

"""
Args:

        params: MDP parameters
        pi: policy that maps the state to action
        p_transition: transition matrix, shape (n_states, n_actions, n_states)
                    At index (state_from, action, state_to)
                    probability P(state_to | state_from, action)
        discount_factor: discount factor [0,1)
        state_values_pi: state values vector, shape(n_states,)
        delta: threshold that checks if the state values converge
        alpha: 1/temperature of boltzmann distribution, larger value makes policy close to the greedy

"""

def compute_v_for_pi(reward, params: mdp_int.IMDPParams[_S, _A], pi: np.ndarray, p_transition: np.ndarray,  discount_factor: float, 
                    state_values_pi = None, delta = 1e-4) -> np.ndarray:
    
    """Returns the state values, i.e. V(s) of the Bellman equation"""
    
    p_transition_pi = np.concatenate([np.expand_dims(p_transition[s,pi[s]], axis=0) for s in range(params.n_states)])   
    state_values_pi = copy.deepcopy(reward) if state_values_pi is None else state_values_pi
    
    while True:
        max_delta = 0
        for s in range(params.n_states):
            old_vs = state_values_pi[s]
            state_values_pi[s] = reward[s] + p_transition_pi[s].dot(discount_factor * state_values_pi)
            max_delta = max(abs(state_values_pi[s] - old_vs), max_delta)
        
        if max_delta < delta:
            return state_values_pi

def __compute_q_s_with_v(reward: np.ndarray, s: int, state_values: np.ndarray, discount_factor: float, p_transition: np.ndarray) -> np.ndarray:
    """ Return the vector result of Q(s,) function for a certain state s"""
    return reward[s] + discount_factor * np.sum(p_transition[s] * state_values, axis = -1)


def __compute_q_with_v(state_values: np.ndarray, discount_factor: float, params: mdp_int.IMDPParams[_S, _A], 
                        reward: np.ndarray, p_transition: np.ndarray):

    """ Return the matrix of Q(s,a) function in Bellman function with a shape (n_states, n_actions)"""

    return np.concatenate([np.expand_dims(__compute_q_s_with_v(reward, s, state_values, discount_factor, p_transition), axis=0)
                               for s in range(params.n_states)])


def compute_q_for_pi(reward, params: mdp_int.IMDPParams[_S, _A], pi, p_transition, discount_factor):
    """ Integrate the calculation of V(s) and Q(s,a) in Bellman functions into a single function"""
    state_values = compute_v_for_pi(reward, 
                                    params, 
                                    pi,
                                    p_transition=p_transition, 
                                    discount_factor=discount_factor, 
                                    state_values_pi = None, 
                                    delta = 1e-4)

    return __compute_q_with_v(state_values, discount_factor, params, reward, p_transition)


def policy_iteration(params: mdp_int.IMDPParams[_S, _A], discount_factor: float, reward: np.ndarray, 
                    p_transition: np.ndarray, pi = None) -> np.ndarray:
    """ return the optimal policy"""
    if pi is None:
        pi = np.random.randint(params.n_actions, size=params.n_states)

    n_iter = 0
    state_values = copy.deepcopy(reward)
    while True:
        old_pi = copy.deepcopy(pi)
        state_values = compute_v_for_pi(reward, 
                                        params, 
                                        pi,
                                        p_transition, 
                                        discount_factor = discount_factor, 
                                        state_values_pi = None, 
                                        delta = 1e-4)

        pi = np.argmax(__compute_q_with_v(state_values, discount_factor, params, reward, p_transition), axis=1)
        if np.all(old_pi == pi):
            return pi
        else:
            n_iter += 1
            if n_iter > 1000:
                print('n_iter: ', n_iter)
                print('rewards: ', reward)


def mcmc_reward(reward: np.ndarray, prior: list, step_size = 0.1):
    """ MCMC step updates the reward """

    new_reward = copy.deepcopy(reward)
    index = np.random.randint(len(reward))
    step = np.random.choice([-step_size, step_size])
    new_reward[index] += step

    if prior[0] == uniform:
        new_reward = np.clip(a=new_reward, a_min=prior[1], a_max=prior[2])

    if np.all(new_reward == reward):
        new_reward[index] -= step

    assert np.any(reward != new_reward), 'rewards do not change: {}, {}'.format(new_reward, reward)
    return new_reward


def compute_posterior(trajectories: Iterable[common.Trajectory[_S, _A]], reward: np.ndarray, 
                    params: mdp_int.IMDPParams[_S, _A], pi: np.ndarray, p_transition: np.ndarray, 
                    discount_factor: float, alpha: int, prior = None):

    """ Compute the log posterior probability P(Reward|Trajectories) """

    q = compute_q_for_pi(reward, params, pi, p_transition, discount_factor,)
    map_states = en.Enumerate(params.states)
    map_actions = en.Enumerate(params.actions)
    likelihood = 0

    for trajectory in trajectories:
        for state, action in zip(trajectory.states, trajectory.actions):
            s = map_states.to_index(state)
            a = map_actions.to_index(action)

            likelihood += alpha * q[s, a] - np.log(np.sum(np.exp(alpha * q[s])))

    if prior is None:
        return likelihood

    else:
        prior_prob = compute_prior(prior=prior[0], reward=reward, min=prior[1],max=prior[2])

        return likelihood + np.log(prior_prob)

def compute_ratio(trajectories: Iterable[common.Trajectory[_S, _A]],
    reward: np.ndarray, 
    params: mdp_int.IMDPParams[_S, _A], 
    pi_tilda: np.ndarray, 
    pi: np.ndarray, 
    alpha: float, 
    discount_factor: float, 
    p_transition: np.ndarray,
    prior=None) -> float:

    """ Compute the ratio of P(New Reward|Trajectory)|P(Reward|Trajectory) """

    ln_p_tilda = compute_posterior(trajectories=trajectories, 
                                    reward=reward, params=params, 
                                    prior=prior, pi=pi_tilda, 
                                    p_transition=p_transition, discount_factor=discount_factor, alpha=alpha)

    ln_p = compute_posterior(trajectories=trajectories, 
                                    reward=reward, params=params, 
                                    prior=prior, pi=pi, 
                                    p_transition=p_transition, discount_factor=discount_factor, alpha=alpha)
    ratio = np.exp(ln_p_tilda - ln_p)

    return ratio

def is_not_optimal(q_values: float, pi: np.ndarray) -> bool:

    """ Detect the changes in optimal policy by checking Q(s,pi(s) < Q(s,a) """
    
    n_states, n_actions = q_values.shape
    for s in range(n_states):
        for a in range(n_actions):
            if q_values[s, pi[s]] < q_values[s, a]:
                return True
    return False

def policy_walk(reward: np.ndarray, 
        discount_factor: float, 
        n_iter:int, 
        params: mdp_int.IMDPParams[_S, _A], 
        p_transition: np.ndarray, 
        trajectories: Iterable[common.Trajectory[_S, _A]], 
        prior: list, 
        alpha=10,
        step_size=0.1):

    """ PolicyWalk Algorithm """

    pi = policy_iteration(params=params,discount_factor=discount_factor,reward=reward, p_transition=p_transition, pi=None)


    for _ in range(n_iter):
        reward_tilda = copy.deepcopy(reward)
        reward_tilda = mcmc_reward(reward=reward, prior=prior, step_size = step_size)
        q_pi_r_tilda = compute_q_for_pi(reward, params, pi, p_transition, discount_factor)

        if is_not_optimal(q_pi_r_tilda, pi):
            pi_tilda = policy_iteration(params = params, discount_factor=discount_factor,reward=reward_tilda, p_transition=p_transition, pi=pi)
            
            if np.random.random() < compute_ratio(trajectories=trajectories, 
                                                reward=reward_tilda, 
                                                pi_tilda=pi_tilda, 
                                                pi=pi, 
                                                p_transition=p_transition, 
                                                discount_factor=discount_factor, 
                                                alpha=alpha,
                                                params=params,
                                                prior=prior):
                
                reward, pi = reward_tilda, pi_tilda
                
    
        else:
            if np.random.random() < compute_ratio(trajectories=trajectories, 
                                                reward=reward_tilda, 
                                                pi_tilda=pi, 
                                                pi=pi, 
                                                p_transition=p_transition, 
                                                discount_factor=discount_factor, 
                                                alpha=alpha,
                                                params=params,
                                                prior=prior):
                reward = reward_tilda
                

    return (reward, pi)