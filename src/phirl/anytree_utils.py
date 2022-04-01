import dataclasses
from typing import Any, Collection, Dict, Iterable, List, Optional, Sequence, Tuple, TypeVar, Union

import anytree
import numpy as np
import pandas as pd


@dataclasses.dataclass
class TreeNaming:
    """Naming conventions used to parse a tree.

    Attrs:
        node: column name with node id/name
        parent: column name with the parent's id
        data: a dictionary mapping column names to field names in nodes

    Example:
    TreeNaming(
        node="Node_ID",
        parent="Parent_ID",
        data={
            "Mutation_ID": "mutation",
            "SomeValue": "value",
        }
    means that a data frame with columns
    "Node_ID", "Parent_ID", "Mutation_ID", "SomeValue"
    is expected.

    Created nodes will have additional fields
    "mutation" and "value".
    """

    node: str = "Node_ID"
    parent: str = "Parent_ID"
    data: Dict[str, str] = dataclasses.field(default_factory=lambda: {"Mutation_ID": "mutation"})


@dataclasses.dataclass
class ForestNaming:
    """Naming conventions used to parse a forest (a set of trees).

    Attrs:
        tree_name: column name storing the tree id/name
        naming: TreeNaming object used to parse each tree
    """

    tree_name: str = "Tree_ID"
    naming: TreeNaming = dataclasses.field(default_factory=TreeNaming)


def parse_tree(df: pd.DataFrame, naming: TreeNaming) -> anytree.Node:
    """Parses a data frame into a tree.

    Args:
        df: data frame with columns specified in `naming`.
        naming: specifies the columns that should be present in `df`

    Returns:
        the root node of the tree
    """
    root = None
    nodes = {}  # Maps a NodeID value to Node

    for _, row in df.iterrows():
        node_id = row[naming.node]
        parent_id = row[naming.parent]
        values = {val: row[key] for key, val in naming.data.items()}

        if node_id in nodes:
            raise ValueError(f"Node {node_id} already exists.")

        # We found the root
        if node_id == parent_id:
            if root is not None:
                raise ValueError(
                    f"Root is {root}, but {node_id} == {parent_id} also looks like a root."
                )
            root = anytree.Node(node_id, parent=None, **values)
            nodes[node_id] = root
        else:
            nodes[node_id] = anytree.Node(node_id, parent=nodes[parent_id], **values)

    return root


def parse_forest(df: pd.DataFrame, naming: ForestNaming) -> Dict[Any, anytree.Node]:
    """Parses a data frame with a forest (a set of trees).

    Args:
        df: data frame with columns specified as in `naming`
        naming: specifies the naming conventions

    Returns:
        dictionary with keys being the tree names (read from the column `naming.tree_name`)
        and values being the root nodes

    See also:
        parse_tree, which powers this function
    """
    result = {}
    for tree_name, tree_df in df.groupby(naming.tree_name):
        result[tree_name] = parse_tree(df=tree_df, naming=naming.naming)

    return result


Trajectory = Tuple[anytree.Node]


def list_all_trajectories(root: anytree.Node, max_length: Optional[int] = None) -> List[Trajectory]:
    """Generates all random walks from `root` downwards.

    Args:
        root: root of the trees
        max_length

    Returns

    """
    if max_length is not None and max_length <= 0:
        raise ValueError(
            f"Max length needs to be at least 1 (was {max_length}). Root node: {root}."
        )

    # If this is leaf, there is only one trajectory.
    if not len(root.children) or max_length == 1:
        return [(root,)]

    ret = []
    for child in root.children:
        mx: Optional[int] = None if max_length is None else max_length - 1
        trajectories = list_all_trajectories(child, max_length=mx)
        for traj in trajectories:
            ret.append(tuple([root] + list(traj)))

    return ret


T = TypeVar("T")


def filter_trajectories(
    trajectories: Iterable[Sequence[T]],
    longer_than: Optional[int] = None,
    shorter_than: Optional[int] = None,
    allowed_length: Union[None, int, Collection[int]] = None,
) -> List[Sequence[T]]:
    """Retains a subset of trajectories which length fulfils given
    constraints.

    Args:
        trajectories: trajectories from which some will be retained
        longer_than: if not None, retains trajectories must have the specified length or be longer
        shorter_than: if not None, retained trajectories must have the specified length
            or be shorter
        allowed_length: if not None, retained trajectories of specified length (or lengths)

    Returns:
        a list of trajectories
    """
    ret = trajectories
    if longer_than is not None:
        ret = filter(lambda x: len(x) >= longer_than, ret)
    if shorter_than is not None:
        ret = filter(lambda x: len(x) <= shorter_than, ret)
    if allowed_length is not None:
        set_lengths = {allowed_length} if isinstance(allowed_length, int) else set(allowed_length)
        ret = filter(lambda x: len(x) in set_lengths, ret)
    return list(ret)


def pick_random_trajectory(trajectories: Collection[Trajectory], seed=None) -> Trajectory:
    """Samples a trajectory from a set. All trajectories are assumed to have equal probability.

    Args:
        trajectories: collection of trajectories
        seed: seed for reproducibility, see np.random.default_rng

    Returns:
        trajectory
    """
    trajectories = list(trajectories)
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(trajectories))
    return trajectories[index]


def construct_random_trajectory(
    root: anytree.Node, max_length: Optional[int] = None, seed=None
) -> Trajectory:
    """Constructs a random trajectory by a random walk from root downwards.

    Args:
        root: root of the tree used to generate a path
        max_length: maximal length of the path. Use None to end at a leaf node.
        seed: seed for reproducibility. See numpy.random.default_rng()

    Returns:
        a list of visited nodes (including `root`)

    Note:
        Longer trajectories have lower probability than the shorter ones.
    """
    rng = np.random.default_rng(seed)

    if max_length is not None and max_length < 1:
        raise ValueError("Path length must be at least 1.")

    # Start the path at the specified root
    path = [root]
    while True:
        # If the path is long enough, end the loop
        if max_length is not None and len(path) >= max_length:
            break
        # If we are at a leaf node, end the loop
        last_visited = path[-1]
        children = last_visited.children
        if len(children) == 0:
            break
        # Select a child at random
        selected = rng.integers(0, len(children))
        selected_child: anytree.Node = children[selected]
        path.append(selected_child)

    return tuple(path)
