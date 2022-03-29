import numpy as np
import numpy.testing as nptest
import pytest

import phirl.maxent as me


@pytest.mark.parametrize("n_actions", (2, 5, 7))
def test_n_states(n_actions: int) -> None:
    n_s = me.get_n_states(n_actions)

    assert isinstance(n_s, int)
    assert n_s == 2**n_actions


@pytest.mark.parametrize("n_actions", (2, 5, 7))
def test_state_space(n_actions: int) -> None:
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


class TestDeterministicTreeMDP:
    @pytest.mark.parametrize("n_actions", (2, 5))
    def test_init(self, n_actions: int) -> None:
        """Tests if the object has proper parameters."""
        mdp = me.DeterministicTreeMDP(n_actions)

        assert mdp.n_states == me.get_n_states(n_actions)
        assert mdp.n_actions == n_actions

    def test_new_state_ok(self) -> None:
        """Tests if the transition function works as expected."""
        n_actions = 3
        mdp = me.DeterministicTreeMDP(n_actions)

        assert mdp.new_state(state=(0, 0, 0), action=1) == (1, 0, 0)
        assert mdp.new_state(state=(1, 1, 0), action=3) == (1, 1, 1)
        # Attempt to do a second mutation on the same gene doesn't
        # change anything
        assert mdp.new_state(state=(1, 0, 0), action=1) == (1, 0, 0)
        assert mdp.new_state(state=(1, 1, 0), action=2) == (1, 1, 0)

    def test_new_state_outside(self, n_actions: int = 5) -> None:
        """Tests whether an exception is raised when we try
        mutations that are disallowed."""
        mdp = me.DeterministicTreeMDP(n_actions)

        for state in mdp.state_space:
            with pytest.raises(ValueError):
                mdp.new_state(state, action=-1)
            with pytest.raises(ValueError):
                mdp.new_state(state, action=0)
            with pytest.raises(ValueError):
                mdp.new_state(state, action=n_actions + 1)
            with pytest.raises(ValueError):
                mdp.new_state(state, action=n_actions + 12)


def featurizer_factory(n_actions: int, featurizer_id: str) -> me.Featurizer:
    """Auxiliary factory function creating featurizer to be tested.

    Args:
        n_actions: number of actions in the spacce
        featurizer_id: either "identity" or "one-hot"

    Returns:
        featurizer
    """
    if featurizer_id == "identity":
        return me.IdentityFeaturizer(n_actions=n_actions)
    elif featurizer_id == "one-hot":
        return me.OneHotFeaturizer(space=me.get_state_space(n_actions))
    else:
        raise ValueError(f"Featurizer type {featurizer_id} now known.")


@pytest.mark.parametrize("n_actions", (2, 3, 5))
@pytest.mark.parametrize("featurizer_id", ("identity", "one-hot"))
def test_featurizer(n_actions: int, featurizer_id: str) -> None:
    """Very basic and generic test checking if featurizers produce
    arrays of the right shape."""
    featurizer = featurizer_factory(n_actions=n_actions, featurizer_id=featurizer_id)

    for state in me.get_state_space(n_actions):
        feature = featurizer.transform(state)
        assert isinstance(feature, np.ndarray)
        assert feature.shape == featurizer.shape


@pytest.mark.parametrize("n_actions", (2, 3, 5))
@pytest.mark.parametrize("featurizer_id", ("identity", "one-hot"))
def test_features_unique(n_actions: int, featurizer_id: str) -> None:
    """Check if the mapping from states to feature vectors
    is injective.

    Note: not all featurizers have this property.
    """
    featurizer = featurizer_factory(n_actions=n_actions, featurizer_id=featurizer_id)
    space = me.get_state_space(n_actions)

    features = set()
    for state in space:
        feature = featurizer.transform(state)
        # Encode feature vector as a tuple, for hashability
        feature_hashable = tuple(feature.ravel().tolist())
        features.add(feature_hashable)

    assert len(features) == len(space)


@pytest.mark.parametrize(
    "state",
    (
        (1, 0, 0),
        (0, 1, 0, 1),
        (1, 0, 0, 1, 0),
        (0, 1),
    ),
)
def test_identity_featurizer(state: me.State) -> None:
    """Check if the identity featurizer works as expected
    on a few simple cases."""
    n_actions = len(state)
    feature = np.asarray(state)

    featurizer = me.IdentityFeaturizer(n_actions=n_actions)
    assert featurizer.shape == feature.shape
    nptest.assert_array_equal(feature, featurizer.transform(state))


class TestOneHotFeaturizer:
    @pytest.mark.parametrize("n_actions", (2, 5))
    def test_inverse(self, n_actions: int) -> None:
        """Check whether the mappings between state and index are inverse to each other."""
        space = me.get_state_space(n_actions)
        featurizer = me.OneHotFeaturizer(space)

        for index, state in enumerate(space):
            assert featurizer.state_to_index(featurizer.index_to_state(index)) == index
            assert featurizer.index_to_state(featurizer.state_to_index(state)) == state

    @pytest.mark.parametrize("n_actions", (2, 5))
    def test_one_hot(self, n_actions: int) -> None:
        space = me.get_state_space(n_actions)
        featurizer = me.OneHotFeaturizer(space)

        for state in space:
            index = featurizer.state_to_index(state)
            nptest.assert_equal(featurizer.transform(state), np.eye(len(space))[index])


class TestExpectedEmpiricalFeatureCounts:
    """Tests for the expected_empirical_feature_counts
    function."""

    def test_2_actions(self) -> None:
        mdp = me.DeterministicTreeMDP(n_actions=2)
        featurizer = me.OneHotFeaturizer(space=me.get_state_space(2))

        trajectories = [
            [(0, 0)],
            [(0, 0), (0, 1)],
            [(0, 0), (1, 0)],
        ]

        feature_counts = me.expected_empirical_feature_counts(
            mdp=mdp, featurizer=featurizer, trajectories=trajectories
        )
        assert isinstance(feature_counts, dict)

        # This state is in all trajectories once
        nptest.assert_equal(feature_counts[(0, 0)], featurizer.transform((0, 0)))
        # These two states appear in 1/3 cases
        nptest.assert_equal(feature_counts[(1, 0)], 1 / 3 * featurizer.transform((1, 0)))
        nptest.assert_equal(feature_counts[(0, 1)], 1 / 3 * featurizer.transform((0, 1)))
        nptest.assert_equal(feature_counts[(1, 1)], np.zeros(featurizer.shape))


class TestTrajectory:
    def test_init_raises(self) -> None:
        with pytest.raises(ValueError):
            me.Trajectory(states=[(0, 1), (0, 0)], actions=[1, 2, 3])
        with pytest.raises(ValueError):
            me.Trajectory(states=[], actions=[2, 3])
        with pytest.raises(ValueError):
            me.Trajectory(states=[(0, 0), (0, 1), (1, 1)], actions=[1])

    def test_init_ok(self) -> None:
        states = ((0, 1), (0, 0), (1, 1))
        actions = (2, 5)
        trajectory = me.Trajectory(states=states, actions=actions)

        assert trajectory.states == states
        assert trajectory.actions == actions

    def test_equality(self) -> None:
        states = ((0, 1), (0, 0), (1, 1))
        actions = (2, 5)
        t1 = me.Trajectory(states=states, actions=actions)
        t2 = me.Trajectory(states=list(states), actions=list(actions))
        assert t1 == t2


def test_unroll_trajectory() -> None:
    mdp = me.DeterministicTreeMDP(n_actions=2)
    states = [(0, 0), (0, 1), (1, 1)]
    actions = [2, 1]
    trajectory = me.Trajectory(
        states=states,
        actions=actions,
    )

    trajectory_ = me.unroll_trajectory(actions=actions, initial_state=(0, 0), mdp=mdp)
    assert trajectory == trajectory_


@pytest.mark.parametrize("n_actions", (5, 10))
def test_unroll_trajectory2(n_actions: int) -> None:
    mdp = me.DeterministicTreeMDP(n_actions=n_actions)
    actions = range(1, n_actions + 1)
    states = [tuple([1] * i + [0] * (n_actions - i)) for i in range(0, n_actions + 1)]
    trajectory = me.Trajectory(states=states, actions=actions)
    trajectory_ = me.unroll_trajectory(actions=actions, initial_state=states[0], mdp=mdp)

    assert trajectory == trajectory_, f"{trajectory} != {trajectory_}"
