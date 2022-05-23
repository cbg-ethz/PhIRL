"""Script running the MaxEnt IRL
algorithm on a trees data set.
Usage:
    python src/run-birl-irl.py data=$PWD/path_to_trees.csv mutations=5

Note:
    The data path must be absolute not relative.
"""
import dataclasses
import logging

import pandas as pd
import numpy as np
import phirl.clustering_birl as birl

import phirl.api as ph
import phirl.hydra_utils as hy
import phirl.sample as sampler
from tqdm import tqdm

@hy.config
@dataclasses.dataclass
class MainConfig:
    """
    Attrs:
        data: absolute path to the CSV with trees
        mutations: the number of mutations
        n_cluters: the number of clusters
        n_iter: the number of iterations in each MCMC step
        alpha: 1/temperature of boltzmann distribution, larger value makes policy close to the greedy
        step_size: 
        naming: settings for parsing the input CSV
        length: settings for filtering out (or truncating) trajectories

    """

    data: str
    mutations: int
    n_clusters: int = 3
    n_iter: int = 100

    discount: float = 0.9
    alpha: float = 10
    step_size: float = 0.1

    naming: ph.ForestNaming = dataclasses.field(default_factory=ph.ForestNaming)
    length: ph.mdp.LengthConfig = dataclasses.field(default_factory=ph.mdp.LengthConfig)


def get_trajectories_each_tree(config: MainConfig, all_trajectories_each_tree):
        action_trajectories_each_tree = [[node.mutation for node in path if node.mutation > 0] for path in all_trajectories_each_tree]
        params = ph.mdp.SimpleParams(n_actions=config.mutations)
        dynamics = ph.mdp.SimpleTransitionFunction(n_actions=config.mutations)

        action_trajectories = ph.mdp.filter_truncate(action_trajectories_each_tree, config=config.length)

        trajectories = [
        ph.mdp.unroll_trajectory(
            actions=actions, initial_state=params.initial_state, dynamics=dynamics
        )
        for actions in action_trajectories
        ]

        return [ph.mdp.add_end_action_and_state(tr) for tr in trajectories]



@hy.main
def main(config: MainConfig) -> None:
    logger = logging.getLogger(__name__)
    logger.info("Starting new run...")
    mdp_params = ph.mdp.EndParams(config.mutations)
    mdp_dynamics = ph.mdp.EndTransitionFunction(config.mutations)

    dataframe = pd.read_csv(config.data)
    trees = ph.parse_forest(dataframe, naming=config.naming)

    # Get the transition probability
    p_t = birl.get_p_transition(params=mdp_params, dynamics=mdp_dynamics)
    prior = ['uniform',-1,1]
    rewards = np.zeros((config.n_clusters,mdp_params.n_states))

    # Initialize the reward in each cluster
    for i in range(config.n_clusters):
        rewards[i] = sampler.sample_reward(params=mdp_params, distribution=prior[0], dist_params=prior[1:3])
    
    membership = {}
    for tree_idx, tree in tqdm(trees.items()):
        # Get trajectories of each tree
        all_trajectories_each_tree = ph.list_all_trajectories(tree)
        trajectories = get_trajectories_each_tree(config, all_trajectories_each_tree)

        # Initialize the cluster index and the corresponding reward
        clusterIdx = sampler.sample_cluster(n_clusters=config.n_clusters)
        #clusterIdx = random.randint(0,2)
        rewards[clusterIdx], pi = birl.policy_walk(reward=rewards[clusterIdx], 
                                discount_factor=config.discount, 
                                n_iter=config.n_iter, 
                                params=mdp_params, 
                                p_transition=p_t, 
                                prior=prior, 
                                trajectories=trajectories,
                                alpha=config.alpha,
                                step_size=config.step_size)
        
        for _ in range(config.n_iter):
        # Sample cluster
            clusterIdx_prime = sampler.sample_cluster(n_clusters=config.n_clusters)
            #clusterIdx_prime = random.randint(0,2)
            reward_prime = rewards[clusterIdx_prime]

            # Update the reward
            reward_prime, pi_prime = birl.policy_walk(reward=reward_prime, 
                                                    discount_factor=config.discount, 
                                                    n_iter=config.n_iter, 
                                                    params=mdp_params, 
                                                    p_transition=p_t, 
                                                    prior=prior, 
                                                    trajectories=trajectories,
                                                    alpha=config.alpha,
                                                    step_size=config.step_size)

            if np.random.random() < birl.compute_ratio(trajectories=trajectories,
                                                    reward=reward_prime,
                                                    params=mdp_params,
                                                    pi_tilda=pi_prime,
                                                    pi=pi,
                                                    alpha=config.alpha,
                                                    discount_factor=config.discount,
                                                    p_transition=p_t):
                
                rewards[clusterIdx] = reward_prime
                clusterIdx = clusterIdx_prime
        membership[tree_idx] = clusterIdx
    
    print(rewards)
    print(membership)
    return membership, rewards



if __name__ == "__main__":
    main()

    