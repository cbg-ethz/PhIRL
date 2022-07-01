"""Script running the MaxEnt IRL
algorithm on a trees data set.
Usage:
    python scripts/run-maxent.py data=$PWD/path_to_trees.csv mutations=5

Note:
    The data path must be absolute not relative.
"""
import dataclasses
import json
import logging
import time
from typing import List

import pandas as pd
import irl_maxent as irl

import phirl.api as ph
import phirl.hydra_utils as hy


@dataclasses.dataclass
class OptimizationConfig:
    lr: float = 0.02
    initial: float = -0.1
    eps: float = 0.02


@hy.config
@dataclasses.dataclass
class MainConfig:
    """
    Attrs:
        data: absolute path to the CSV with trees
        mutations: the number of mutations
        identity: whether to use the identity featurizer
        optimization: settings for the optimization
        naming: settings for parsing the input CSV
        length: settings for filtering out (or truncating) trajectories

        causal: whether to use Maximal *Causal* Entropy IRL
        discount: discount factor, used only if `causal` is True
    """

    data: str
    mutations: int
    identity: bool = True

    causal: bool = False
    discount: float = 0.9

    optimization: OptimizationConfig = dataclasses.field(default_factory=OptimizationConfig)
    naming: ph.ForestNaming = dataclasses.field(default_factory=ph.ForestNaming)
    length: ph.mdp.LengthConfig = dataclasses.field(default_factory=ph.mdp.LengthConfig)


def get_action_trajectories(config: MainConfig) -> List[List[int]]:
    dataframe = pd.read_csv(config.data)
    trees = ph.parse_forest(dataframe, naming=config.naming)

    # Get all paths from every tree and concatenate into one list
    all_paths = sum([ph.list_all_trajectories(tree) for tree in trees.values()], [])

    # Return the mutations (actions) form each node
    return [[node.mutation for node in path if node.mutation > 0] for path in all_paths]


def get_trajectories(config: MainConfig) -> List[ph.mdp.Trajectory]:
    params = ph.mdp.SimpleParams(n_actions=config.mutations)
    dynamics = ph.mdp.SimpleTransitionFunction(n_actions=config.mutations)

    action_trajectories = get_action_trajectories(config)
    action_trajectories = ph.mdp.filter_truncate(action_trajectories, config=config.length)

    trajectories = [
        ph.mdp.unroll_trajectory(
            actions=actions, initial_state=params.initial_state, dynamics=dynamics
        )
        for actions in action_trajectories
    ]
    return trajectories


def get_featurizer(identity: bool, params: ph.mdp.EndParams) -> ph.mdp.EndFeaturizer:
    if identity:
        return ph.mdp.EndIdentityFeaturizer(params)
    else:
        return ph.mdp.EndOneHotFeaturizer(params)


@hy.main
def main(config: MainConfig) -> None:
    time_start = time.time()

    logger = logging.getLogger(__name__)
    logger.info("Starting new run...")

    logger.info("Parsing the trajectories...")
    # Get the "normal" trajectories
    trajectories = get_trajectories(config)
    # Now add the end state
    trajectories = [ph.mdp.add_end_action_and_state(tr) for tr in trajectories]

    # We need to define the MDP with an additional end state
    logger.info("Defining the MDP dynamics...")
    mdp_params = ph.mdp.EndParams(config.mutations)
    mdp_dynamics = ph.mdp.EndTransitionFunction(config.mutations)

    # Initialize optimizer and initial reward weights
    logger.info("Initializing the optimizer...")
    init = irl.optimizer.Constant(config.optimization.initial)
    optim = irl.optimizer.Sga(lr=config.optimization.lr)

    # Process the data so it fits the interface specifications
    logger.info("Calculating the dynamics matrix and the feature matrix...")
    p_t = ph.get_p_transition(params=mdp_params, dynamics=mdp_dynamics)
    features = ph.get_features(
        params=mdp_params, featurizer=get_featurizer(identity=config.identity, params=mdp_params)
    )
    # We have a unique terminal state.
    # TODO(Pawel): Generalize to the setting in which we use the "simple" MDP, but
    #  we manually mark some states as the end ones
    terminal = ph.get_terminal(params=mdp_params, terminal_states=[ph.mdp.END_STATE])
    trajectories_ = ph.get_trajectories(params=mdp_params, trajectories=trajectories)

    logger.info("Finding the rewards...")
    if not config.causal:
        rewards = irl.maxent.irl(
            p_transition=p_t,
            features=features,
            terminal=terminal,
            trajectories=trajectories_,
            optim=optim,
            init=init,
            eps=config.optimization.eps,
        )
    else:
        rewards = irl.maxent.irl_causal(
            p_transition=p_t,
            features=features,
            terminal=terminal,
            trajectories=trajectories_,
            optim=optim,
            init=init,
            eps=config.optimization.eps,
            discount=config.discount,
        )

    logger.info("Saving the results...")
    results = {
        "states": mdp_params.states,
        "features": features.tolist(),
        "rewards": rewards.tolist(),
        "time": time.time() - time_start,
    }

    with open("results.json", "w") as fp:
        json.dump(obj=results, fp=fp)

    logger.info("Run finished.")


if __name__ == "__main__":
    main()
