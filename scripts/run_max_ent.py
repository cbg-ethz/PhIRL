"""Script running the MaxEnt IRL
algorithm on a trees data set.
Usage:
    python scripts/run_max_ent.py data=$PWD/path_to_trees.csv n_action=5
Note:
    The data path must be absolute not relative.
"""
import dataclasses
import enum
import logging
from typing import Any, Dict, Optional

import anytree
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import phirl.api as ph
import phirl.hydra_utils as hy
import phirl.mdp as me
from irl_maxent import optimizer


class Featurizer(enum.Enum):
    """Enum used to select featurizer to be used."""

    ONEHOT = "ONEHOT"
    IDENTITY = "IDENTITY"


@hy.config
@dataclasses.dataclass
class MainConfig:
    """Class storing arguments for the script.
    Attrs:
        data: path to a data frame with trees
        n_action: number of actions (mutations)
        learning_rate: the learning rate used for gradient descent optimization
        eps: the threshold of convergence for optimization algorithm
        max_iter: maximum number of iterations in the optimization loop
        naming: conventions used to name forest
        max_length: use to limit the maximal length of the trajectory
    """

    data: str
    n_action: int = 5
    learning_rate: float = 0.05
    featurizer: Featurizer = Featurizer.IDENTITY
    eps: float = 1e-2
    max_iter: int = 1000

    naming: ph.ForestNaming = dataclasses.field(default_factory=ph.ForestNaming)
    max_length: Optional[int] = None


def get_trees(config: MainConfig) -> Dict[Any, anytree.Node]:
    """This function parses the original data into readable tree data structure"""
    dataframe = pd.read_csv(config.data)
    trees = ph.parse_forest(dataframe, naming=config.naming)
    return trees


def get_featurizer(config: MainConfig) -> me.Featurizer:
    """Factory method creating the featurizer specified in the config."""
    if config.featurizer == Featurizer.ONEHOT:
        space = me.get_state_space(n_actions=config.n_action)
        return me.OneHotFeaturizer(space)
    elif config.featurizer == Featurizer.IDENTITY:
        return me.IdentityFeaturizer(n_actions=config.n_action)
    else:
        raise ValueError(f"Featurizer {config.featurizer} not recognized.")


def save_results(results: me.IRLOutput) -> None:
    """This function saves the results."""
    # Use the fact that Hydra substitutes the working directory
    me.save_output(results, path=".")

    # Draw the plot
    panel_size = 5
    fig, axs = plt.subplots(2, figsize=(panel_size, panel_size * 2))

    def plot_history(ax: plt.Axes, values) -> None:
        y = list(values)
        x = (range(1, len(y) + 1),)
        ax.scatter(x, y)

    axs[0].set_title("Grad norm")
    plot_history(axs[0], [np.linalg.norm(entry.grad) for entry in results.history])

    axs[1].set_title("Theta difference norm")
    theta_hist = np.asarray([entry.theta for entry in results.history])
    theta_diff = theta_hist[1:, ...] - theta_hist[:-1, ...]
    plot_history(axs[1], map(np.linalg.norm, theta_diff))

    fig.tight_layout()
    fig.savefig("plot.pdf")


@hy.main
def main(config: MainConfig) -> None:
    logger = logging.getLogger(__name__)
    logger.info("Starting new run...")

    logger.info(f"Creating MDP with {config.n_action} actions...")
    state_space = me.get_state_space(config.n_action)
    mdp = ph.maxent.DeterministicTreeMDP(config.n_action)

    logger.info(f"Reading data from {config.data} file...")
    trees = get_trees(config)

    logger.info("Setting up featurizer...")
    featurizer = get_featurizer(config)
    features = me.get_features(featurizer, state_space)
    logger.info(f"Feature dimensionality: {featurizer.shape}")

    logger.info("Extracting trajectories...")
    action_of_trajectories = me.get_action_of_trajectories(trees, max_length=20)
    initial_state = me.initial_state(config.n_action)

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
    optim = optimizer.Sga(lr=config.learning_rate)

    results = me.irl(
        n_actions=config.n_action,
        features=features,
        feature_expectation=feature_expectation,
        optim=optim,
        mdp=mdp,
        eps=config.eps,
        max_iter=config.max_iter,
    )

    logger.info("Saving the results...")
    save_results(results)


if __name__ == "__main__":
    main()
