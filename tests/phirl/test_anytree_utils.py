import anytree
import pandas as pd

import phirl.api as ph


def test_parse_tree() -> None:
    """Parse a simple tree with root 1 and two branches:
     - 1 -> 2
     - 1 -> 5 -> 10
    Each node has two values: n**2 and 2*n, where n is its node id.
    """
    tree_df = pd.DataFrame(
        {
            "NodeID": [1, 2, 5, 10],
            "ParentID": [1, 1, 1, 5],
            "ValueSquare": [1, 4, 25, 100],
            "ValueDoubled": [2, 4, 10, 20],
        }
    )
    naming = ph.TreeNaming(
        node="NodeID",
        parent="ParentID",
        values={
            "ValueSquare": "square",
            "ValueDoubled": "doubled",
        },
    )
    root = ph.parse_tree(df=tree_df, naming=naming)

    for node in anytree.PreOrderIter(root):
        n = node.name
        assert node.square == n**2
        assert node.doubled == 2 * n

        if n == 2 or n == 5:
            assert node.parent.name == 1
        elif n == 10:
            assert node.parent.name == 5
