import dataclasses
from typing import Any, Dict

import anytree
import pandas as pd


@dataclasses.dataclass
class TreeNaming:
    """Naming conventions used to parse a tree.

    Attrs:
        node: column name with node id/name
        parent: column name with the parent's id
        values: a dictionary mapping

    Example:
    TreeNaming(
        node="Node_ID",
        parent="Parent_ID",
        values={
            "Mutation_ID": "mutation",
            "SomeValue": value,
        }
    means that a data frame with columns
    "Node_ID", "Parent_ID", "Mutation_ID", "SomeValue"
    is expected.

    Created nodes will have additional fields
    "mutation" and "value".
    """

    node: str = "Node_ID"
    parent: str = "Parent_ID"
    values: Dict[str, str] = dataclasses.field(default_factory=lambda: {"Mutation_ID": "mutation"})


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
        values = {val: row[key] for key, val in naming.values.items()}

        if node_id in nodes:
            raise ValueError(f"Node {node_id} already exists.")

        # We found the root
        if node_id == parent_id:
            if root is not None:
                raise ValueError(
                    f"Root is {root}, but {node_id} == {parent_id} also looks like a root."
                )
            root = anytree.Node(node_id, **values)
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
