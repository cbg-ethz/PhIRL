from typing import List

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
        assert node.square == n**2
        assert node.doubled == 2 * n

        if n == 2 or n == 5:
            assert node.parent.name == 1
        elif n == 10:
            assert node.parent.name == 5


def test_parse_forest() -> None:
    forest = pd.DataFrame(
        {
            "Patient_ID": [1, 1, 2, 2, 2, 3, 3, 3, 3, 3],
            "Tree_ID": [1, 1, 2, 2, 2, 3, 3, 3, 3, 3],
            "Node_ID": [1, 2, 1, 2, 3, 1, 2, 3, 4, 5],
            "Mutation_ID": [0, 4, 0, 4, 3, 0, 2, 4, 3, 4],
            "Parent_ID": [1, 1, 1, 1, 2, 1, 1, 1, 2, 2],
        }
    )
    naming = ph.ForestNaming(
        tree_name="Tree_ID",
        naming=ph.TreeNaming(
            node="Node_ID",
            parent="Parent_ID",
            values={
                "Mutation_ID": "mutation",
            },
        ),
    )

    parsed = ph.parse_forest(forest, naming=naming)
    assert len(parsed) == 3
    for root in parsed.values():
        assert isinstance(root, anytree.Node)
        assert hasattr(root, "mutation")


@pytest.fixture
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


@pytest.fixture
def simple_trajectories(simple_tree: anytree.Node) -> List[ph.Trajectory]:
    return ph.list_all_trajectories(simple_tree)


def trajectory_names(traj: ph.Trajectory):
    return tuple(node.name for node in traj)


@pytest.mark.parametrize("max_length", (1, 2, 3, 10))
def test_list_all_trajectories(simple_tree: anytree.Node, max_length: int) -> None:
    trajectories = ph.list_all_trajectories(simple_tree, max_length=max_length)
    traj = {trajectory_names(t) for t in trajectories}

    print(traj)  # The output is suppressed if no errors are present
    if max_length == 1:
        assert len(traj) == 1
        assert (1,) in traj
    elif max_length == 2:
        assert len(traj) == 2
        assert (1, 2) in traj
        assert (1, 3) in traj
    elif max_length >= 3:
        assert len(traj) == 3
        assert (1, 3) in traj
        assert (1, 2, 4) in traj
        assert (1, 2, 5) in traj

    assert ph.list_all_trajectories(simple_tree, max_length=3) == ph.list_all_trajectories(
        simple_tree, max_length=None
    )


@pytest.mark.parametrize("max_length", (0, -1, -5))
def test_list_all_trajectories_negative(simple_tree: anytree.Node, max_length: int) -> None:
    with pytest.raises(ValueError):
        ph.list_all_trajectories(simple_tree, max_length=max_length)


def test_pick_random_trajectory_reproducible(simple_trajectories) -> None:
    t1 = ph.pick_random_trajectory(simple_trajectories, seed=0)
    t2 = ph.pick_random_trajectory(simple_trajectories, seed=0)
    assert t1 == t2


def test_pick_random_trajectory_different(simple_trajectories) -> None:
    rng = np.random.default_rng(42)
    t1 = ph.pick_random_trajectory(simple_trajectories, seed=rng)
    t2 = ph.pick_random_trajectory(simple_trajectories, seed=rng)
    # Chance that they are equal should be 1/3. Fortunately, this is the case for seed 42.
    assert t1 != t2


def test_construct_random_trajectory_reproducible(simple_tree: anytree.Node) -> None:
    t1 = ph.construct_random_trajectory(simple_tree, seed=0)
    t2 = ph.construct_random_trajectory(simple_tree, seed=0)
    assert t1 == t2


@pytest.mark.parametrize("max_length", (1, 2, 3, 10))
def test_construct_random_trajectory_all(simple_tree: anytree.Node, max_length: int) -> None:
    rng = np.random.default_rng(111)

    trajs = set()
    for _ in range(20):
        trajs.add(ph.construct_random_trajectory(simple_tree, seed=rng, max_length=max_length))
    assert trajs == set(ph.list_all_trajectories(simple_tree, max_length=max_length))


def test_filter_trajectories() -> None:
    # The type is different, but it shouldn't matter in this case
    # thanks to duck-typing
    traj = [
        (1,),
        (3, 4),
        (1, 2, 3),
        (4, 5, 5),
        (5, 6, 7, 8),
    ]

    assert ph.filter_trajectories(traj, longer_than=2) == traj[1:]
    assert ph.filter_trajectories(traj, longer_than=2, shorter_than=3) == traj[1:4]
    assert ph.filter_trajectories(traj, allowed_length=1) == traj[:1]
    assert ph.filter_trajectories(traj, allowed_length=(1, 4)) == [traj[0], traj[4]]
