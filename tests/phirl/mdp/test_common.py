import pytest

import phirl.mdp.common as common
import phirl.mdp.simple as simple


class TestTrajectory:
    def test_init_raises(self) -> None:
        with pytest.raises(ValueError):
            common.Trajectory(states=[(0, 1), (0, 0)], actions=[1, 2, 3])
        with pytest.raises(ValueError):
            common.Trajectory(states=[], actions=[2, 3])
        with pytest.raises(ValueError):
            common.Trajectory(states=[(0, 0), (0, 1), (1, 1)], actions=[1])

    def test_init_ok(self) -> None:
        states = ((0, 1), (0, 0), (1, 1))
        actions = (2, 5)
        trajectory = common.Trajectory(states=states, actions=actions)

        assert trajectory.states == states
        assert trajectory.actions == actions

    def test_equality(self) -> None:
        states = ((0, 1), (0, 0), (1, 1))
        actions = (2, 5)
        t1 = common.Trajectory(states=states, actions=actions)
        t2 = common.Trajectory(states=list(states), actions=list(actions))
        assert t1 == t2


def test_unroll_trajectory() -> None:
    states = [(0, 0), (0, 1), (1, 1)]
    actions = [2, 1]
    trajectory = common.Trajectory(
        states=states,
        actions=actions,
    )

    dynamics = simple.SimpleTransitionFunction(n_actions=2)
    trajectory_ = common.unroll_trajectory(actions=actions, initial_state=(0, 0), dynamics=dynamics)

    assert trajectory == trajectory_
