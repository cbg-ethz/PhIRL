"""Script running the MaxEnt IRL
algorithm on a trees data set.
Usage:
    python scripts/run_max_ent.py data=$PWD/path_to_trees.csv n_action=5
Note:
    The data path must be absolute not relative.
"""
import dataclasses
from locale import normalize
import logging
from typing import Any, Dict, Optional

import anytree
import pandas as pd
import numpy as np

import phirl.api as ph
import phirl.hydra_utils as hy
from phirl.maxent import *
import irl_maxent as me


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
    logger.info("Starting new run for states...")

    logger.info(f"Creating MDP with {config.n_action} actions...")
    SP = State_space(config.n_action)
    mdp = ph.maxent.DeterministicTreeMDP(n_actions=config.n_action, SP=State_space(config.n_action))

    logger.info(f"Reading data from {config.data} file...")
    trees = get_trees(config)
    n_states = get_n_states(config.n_action)
    state_space = SP.get_state_space()

    TS = StateTransitions(config.n_action, trees, SP)
    #p_transition = TS.get_p_transition()
    transitions = TS.get_transition()
    features = get_features(IdentityFeaturizer(Featurizer), state_space)
    #logger.info(f"Features: {features}")

    logger.info("Calculating feature expectations...")
    # featurizer = OneHotFeaturizer(state_space)
    counts, feature_expectation = expected_empirical_feature_counts_from_trajectories(mdp, featurizer=IdentityFeaturizer(Featurizer), trajectories=transitions)
    #d = expected_svf_from_policy(p_transition, p_action, eps=1e-5)
    
    logger.info("Learning the state reward function...")
    optim = me.optimizer.ExpSga(lr=me.optimizer.linear_decay(lr0=0.2), normalize=True)
    p_initial = np.zeros(n_states)
    p_initial[0] = 1
    irl = IRL(features, feature_expectation, optim, TS, p_initial, eps=1e-4, eps_esvf=1e-5)
    s_reward = irl.irl_state()

    logger.info("Learning the action reward function...")
    action_space = SP.get_action_space()
    action_features = get_features(IdentityFeaturizer(Featurizer), action_space)
    a_reward = irl.irl_action(action_features)
    s_a_reward = irl.additive_reward(config.n_action, action_features)
    
    logger.info(f"Feature expectation: {feature_expectation}")
    logger.info(f"Feature dimension: {len(features[1])}")
    logger.info(f"State reward function: {s_reward}")
    logger.info(f"Action reward function: {a_reward}")
    logger.info(f"Additive reward function: {s_a_reward}")

# ************************************************************************


if __name__ == "__main__":
    main()