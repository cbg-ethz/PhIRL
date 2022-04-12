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
#from matplotlib.pyplot import get
import pandas as pd
import numpy as np

import phirl.api as ph
import phirl.hydra_utils as hy
import phirl.maxent as me
from irl_maxent import optimizer as Optim
import seaborn as sns
import matplotlib.pyplot as plt

# import optimizer as Optim


# ************************************************************************
@hy.config
@dataclasses.dataclass
class MainConfig:
    """Class storing arguments for the script.
    Attrs:
        data: path to a data frame with trees
        n_action: number of actions (mutations)
        learning_rate: the learning rate used for gradient descent optimization
        eps: the threshold of convergence for optimization algorithm
        eps_esvf: The threshold to be used as convergence criterion for the expected state-visitation frequency.
        naming: conventions used to name forest
        max_length: use to limit the maximal length of the trajectory
    """

    data: str = "/Users/apple/Desktop/Lab_rotation1/tree_df.csv"
    n_action: int = 5
    learning_rate: float = 0.5
    eps: float = 1e-2

    naming: ph.ForestNaming = dataclasses.field(default_factory=ph.ForestNaming)
    max_length: Optional[int] = None


def get_trees(config: MainConfig) -> Dict[Any, anytree.Node]:
    """This function parses the original data into readable tree data structure"""
    dataframe = pd.read_csv(config.data)
    trees = ph.parse_forest(dataframe, naming=config.naming)
    return trees


@hy.main
def main(config: MainConfig) -> None:
    logger = logging.getLogger(__name__)
    logger.info("Starting new run for states...")

    logger.info(f"Creating MDP with {config.n_action} actions...")
    state_space = me.get_state_space(config.n_action)
    mdp = ph.maxent.DeterministicTreeMDP(config.n_action)

    logger.info(f"Reading data from {config.data} file...")
    trees = get_trees(config)

    logger.info("Setting up featurizier...")
    # featurizer = me.IdentityFeaturizer(config.n_action)
    featurizer = me.OneHotFeaturizer(state_space)
    features = me.get_features(featurizer, state_space)
    logger.info(f"Feature dimensionality: {featurizer.shape}")

    logger.info("Extracting trajectories...")
    action_of_trajectories = me.get_action_of_trajectories(trees, max_length=20)
    n_states = me.get_n_states(config.n_action)
    initial_state = tuple(0 for _ in range(config.n_action))
    trajectories = me.unroll_trajectories(
        action_trajectories=action_of_trajectories, initial_state=initial_state, mdp=mdp
    )
    state_trajectories = me.get_state_transition_trajectories(trajectories)

    logger.info("Calculating feature expectations...")
    counts = me.expected_empirical_feature_counts(
        mdp=mdp, featurizer=featurizer, trajectories=state_trajectories
    )
    feature_expectation = sum(np.array(list(counts.values())))
    logger.info(f"Feature expectation: {feature_expectation}")

    logger.info("Learning the state reward function...")
    optim = Optim.Sga(lr=config.learning_rate)
    # optim = Optim.ExpSga(lr=Optim.linear_decay(lr0=config.learning_rate), normalize=True)
    # optim = Optim.NormalizeGrad(optim)

    s_reward, _, learning_history, theta_history = me.irl(
        n_actions=config.n_action,
        features=features,
        feature_expectation=feature_expectation,
        optim=optim,
        eps=config.eps,
        mdp=mdp,
    )
    logger.info(f"State reward function: {s_reward}")

    # additive_reward = me.get_additive_reward(n_actions=config.n_action, learned_reward=s_reward)
    # logger.info(f"Additive reward function: {additive_reward}")

    # action_reward = me.get_action_reward(n_actions = config.n_action, learned_reward = s_reward)
    # logger.info(f"One-action state reward function: {action_reward}")

    fig = me.plot_learning_history(
        learning_history=learning_history, theta_history=theta_history
    )
    # fig.savefig("plot.pdf")


# ************************************************************************


if __name__ == "__main__":
    main()
