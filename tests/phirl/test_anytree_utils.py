from typing import Set

import anytree
import numpy as np
import pandas as pd
import pytest

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
        assert node.square == n ** 2
        assert node.doubled == 2 * n

        if n == 2 or n == 5:
            assert node.parent.name == 1
        elif n == 10:
            assert node.parent.name == 5


class TestTreeRandomWalk:
    """Tests for the tree_random_walk function."""

    @staticmethod
    def simple_tree() -> anytree.Node:
        """A simple tree with 3 paths:
            1
           / \
          3   2
             / \
            4   5
        """
        n1 = anytree.Node(1, parent=None)
        n2 = anytree.Node(2, parent=n1)
        anytree.Node(3, parent=n1)
        anytree.Node(4, parent=n2)
        anytree.Node(5, parent=n2)
        return n1

    @staticmethod
    def get_paths(root: anytree.Node, max_length, n_paths: int) -> Set[tuple]:
        rng = np.random.default_rng(42)
        paths = set()
        for _ in range(n_paths):
            path = ph.tree_random_walk(root, max_length=max_length, seed=rng)
            paths.add(tuple(node.name for node in path))
        return paths

    def traverse_all_paths(self) -> None:
        root = self.simple_tree()
        # Generate 10 random paths
        paths = self.get_paths(root, max_length=None, n_paths=10)
        # There are only 3 unique random walks possible
        assert len(paths) == 3
        assert (1, 2, 4) in paths
        assert (1, 2, 5) in paths
        assert (1, 3) in paths

    def test_max_length_specified_1(self) -> None:
        root = self.simple_tree()
        paths = self.get_paths(root, max_length=1, n_paths=5)
        assert len(paths) == 1
        assert (1,) in paths

    def test_max_length_specified_2(self) -> None:
        root = self.simple_tree()
        paths = self.get_paths(root, max_length=2, n_paths=5)

        assert len(paths) == 2
        assert (1, 2) in paths
        assert (1, 3) in paths

    @pytest.mark.parametrize("max_length", (0, -1, -5))
    def test_max_length_negative(self, max_length: int) -> None:
        """Check if a ValueError is raised for nonsensical `max_length`."""
        with pytest.raises(ValueError):
            ph.tree_random_walk(anytree.Node(1), max_length=max_length, seed=0)
