"""Script running the MaxEnt IRL
algorithm on a trees data set.
Usage:
    python scripts/run-maxent.py data=$PWD/path_to_trees.csv mutations=5

Note:
    The data path must be absolute not relative.
"""
import dataclasses
import logging
from typing import List

import pandas as pd
import irl_maxent as irl

import phirl.api as ph
import phirl.hydra_utils as hy


@dataclasses.dataclass
class OptimizationConfig:
    lr: float = 0.05
    initial: float = 0.5
    eps: float = 0.05


@hy.config
@dataclasses.dataclass
class MainConfig:
    data: str
    mutations: int

    optimization: OptimizationConfig = dataclasses.field(default_factory=OptimizationConfig)
    naming: ph.ForestNaming = dataclasses.field(default_factory=ph.ForestNaming)
    length: ph.mdp.LengthConfig = dataclasses.field(default_factory=ph.mdp.LengthConfig)


def get_action_trajectories(config: MainConfig) -> List[List[int]]:
    dataframe = pd.read_csv(config.data)
    trees = ph.parse_forest(dataframe, naming=config.naming)

    # Get all paths from every tree and concatenate into one list
    all_paths = sum([ph.list_all_trajectories(tree) for tree in trees], start=[])

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


@hy.main
def main(config: MainConfig) -> None:
    logger = logging.getLogger(__name__)
    logger.info("Starting new run...")

    logger.info("Parsing the trajectories...")
    # Get the "normal" trajectories
    trajectories = get_trajectories(config)
    # Now add the end state
    trajectories = [ph.mdp.add_end_action_and_state(tr) for tr in trajectories]

    # We need to define the MDP with an additional end state
    mdp_params = ph.mdp.EndParams(config.mutations)
    mdp_dynamics = ph.mdp.EndTransitionFunction(config.mutations)

    # Initialize optimizer and initial reward weights
    init = irl.optimizer.Constant(config.optimization.initial)
    optim = irl.optimizer.Sga(lr=config.optimization.lr)

    x = irl.maxent.irl(
        p_transition=ph.get_p_transition(params=mdp_params, dynamics=mdp_dynamics),
        features=ph.get_features(
            params=mdp_params, featurizer=ph.mdp.EndIdentityFeaturizer(params=mdp_params)
        ),
        terminal=ph.get_terminal(params=mdp_params, terminal_states=[ph.mdp.END_STATE]),
        trajectories=ph.get_trajectories(params=mdp_params, trajectories=trajectories),
        optim=optim,
        init=init,
        eps=config.optimization.eps,
    )

    logger.info(x)
    logger.info("Run finished.")


if __name__ == "__main__":
    main()
