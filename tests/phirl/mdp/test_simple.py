import pytest

import phirl.mdp.simple as simple


class TestSimpleParams:
    @pytest.mark.parametrize("n_actions", (2, 3, 5))
    def test_n_states_n_actions(self, n_actions: int) -> None:
        params = simple.SimpleParams(n_actions)

        assert params.n_actions == n_actions
        assert params.n_states == 2**n_actions

    @pytest.mark.parametrize("n_actions", (2, 3, 5))
    def test_state_space(self, n_actions: int) -> None:
        params = simple.SimpleParams(n_actions)

        assert len(params.states) == params.n_states, "State space has wrong cardinality"
        assert len(set(params.states)) == params.n_states, "States not unique"

        for state in params.states:
            assert isinstance(state, tuple)
            assert len(state) == n_actions
            assert min(state) >= 0
            assert min(state) <= 1

    @pytest.mark.parametrize(
        "params", [(1, (0,)), (3, (0, 0, 0)), (10, tuple(0 for _ in range(10)))]
    )
    def test_initial_state(self, params) -> None:
        n_actions, answer = params
        assert simple.SimpleParams(n_actions).initial_state == answer
