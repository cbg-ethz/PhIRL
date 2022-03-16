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

    TS = StateTransitions(config.n_action, trees)
    #p_transition = TS.get_p_transition()
    transitions = TS.get_transition()
    features = get_features(IdentityFeaturizer(Featurizer), state_space)

    logger.info("Calculating feature expectations...")
    # featuerizer = OneHotFeaturizer(state_space)
    counts, feature_expectation = expected_empirical_feature_counts_from_trajectories(mdp, featurizer=IdentityFeaturizer(), trajectories=transitions)
    #d = expected_svf_from_policy(p_transition, p_action, eps=1e-5)
    logger.info("Learning the reward function...")
    optim = Sga(lr=1e-4)
    reward = irl(features, feature_expectation, optim, TS, eps=1e-4, eps_esvf=1e-5)
    
    #logger.info(f"Feature expectation: {feature_expectation}")
    #logger.info(f"Expected SVF: {d}")
    #logger.info(f"Local action probabilities: {p_action}")
    #logger.info(f"Local transition probabilities: {p_transition}")
    logger.info(f"Reward function: {reward}")


if __name__ == "__main__":
    main()