import pytest

import phirl.maxent as me


@pytest.mark.parametrize("n_actions", (2, 5, 7))
def test_n_states(n_actions: int) -> None:
    n_s = me.get_n_states(n_actions)

    assert isinstance(n_s, int)
    assert n_s == 2 ** n_actions


@pytest.mark.parametrize("n_actions", (2, 5, 7))
def test_state_space(n_actions) -> None:
    space = me.get_state_space(n_actions)
    size = me.get_n_states(n_actions)

    # Check the size of the space
    assert len(space) == size

    # Check whether the states are unique
    assert len(set(space)) == size

    # Check each state separately
    for state in space:
        assert isinstance(state, tuple)
        assert len(state) == n_actions

        # Check if this is a binary vector
        assert max(state) <= 1
        assert min(state) >= 0
