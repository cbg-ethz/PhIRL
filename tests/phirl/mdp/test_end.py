import pytest

import phirl.mdp.end as end


class TestEndParams:
    @pytest.mark.parametrize("n", (2, 3, 5))
    def test_n_actions_n_states(self, n: int) -> None:
        params = end.EndParams(n)

        assert params.n_actions == n + 1
        assert params.n_states == 2**n + 1
