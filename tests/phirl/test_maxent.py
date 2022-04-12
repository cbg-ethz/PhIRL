from typing import List, Tuple
import numpy as np
import numpy.testing as nptest
from phirl.anytree_utils import Trajectory
import pytest
import anytree

import phirl.maxent as me
import phirl.api as ph
from irl_maxent import optimizer as O


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


@pytest.mark.parametrize("n_actions", (2, 5, 7))
def test_get_features(n_actions) -> None:
    """Tests for the function that mapping the state to its corresponding features"""
    featurizer = me.IdentityFeaturizer(n_actions)
    state_space = me.get_state_space(n_actions)

    features = me.get_features(featurizer=featurizer, state_space=state_space)

    assert features.shape == (len(state_space), n_actions)
    assert features.max() == 1
    assert features.min() == 0


class TestGetPTransition:
    """Tests for function calculating the probability of transition"""

    def test_3_mutation(self, n_actions=3) -> None:
        mdp = me.DeterministicTreeMDP(n_actions)
        state_space = me.get_state_space(n_actions)
        p_transition = me.get_p_transition(n_actions, state_space=state_space, mdp=mdp)
        current_state1 = state_space.index(tuple([0, 0, 0]))
        next_state1 = state_space.index(tuple([1, 0, 0]))
        action1 = 0

        current_state2 = state_space.index(tuple([1, 0, 0]))
        next_state2 = state_space.index(tuple([1, 0, 1]))
        action2 = 2

        current_state3 = state_space.index(tuple([1, 0, 0]))
        next_state3 = state_space.index(tuple([1, 0, 1]))
        action3 = 1

        nptest.assert_equal(p_transition[current_state1, next_state1, action1], 1)
        nptest.assert_equal(p_transition[current_state2, next_state2, action2], 1)
        nptest.assert_equal(p_transition[current_state3, next_state3, action3], 0)

    def test_6_mutation(self, n_actions=6) -> None:

        mdp = me.DeterministicTreeMDP(n_actions)
        state_space = me.get_state_space(n_actions)
        p_transition = me.get_p_transition(n_actions, state_space=state_space, mdp=mdp)
        current_state1 = state_space.index(tuple([0, 0, 0, 0, 0, 0]))
        next_state1 = state_space.index(tuple([0, 0, 0, 0, 0, 1]))
        action1 = 5

        current_state2 = state_space.index(tuple([1, 1, 1, 0, 0, 0]))
        next_state2 = state_space.index(tuple([0, 0, 0, 0, 0, 1]))
        action2 = 0

        current_state3 = state_space.index(tuple([1, 1, 1, 0, 0, 0]))
        next_state3 = state_space.index(tuple([1, 1, 1, 1, 0, 0]))
        action3 = 3

        nptest.assert_equal(p_transition[current_state1, next_state1, action1], 1)
        nptest.assert_equal(p_transition[current_state2, next_state2, action2], 0)
        nptest.assert_equal(p_transition[current_state3, next_state3, action3], 1)


class TestPAction:
    """Tests for function calculating the local probability of action"""

    def test_p_action_2_action(self, n_actions=2) -> None:
        state_space = me.get_state_space(n_actions)
        n_states = me.get_n_states(n_actions)
        reward = np.zeros(n_states) + 1
        p_action = me.get_p_action(n_states, n_actions, reward, state_space)

        assert p_action[0, 1] == 1 / 2
        assert p_action[0, 0] == 1 / 2
        assert p_action[1, 0] == 1
        assert p_action[2, 1] == 1

    def test_p_action_4_action(self, n_actions=4) -> None:
        state_space = me.get_state_space(n_actions)
        n_states = me.get_n_states(n_actions)
        reward = np.zeros(n_states) + 1
        p_action = me.get_p_action(n_states, n_actions, reward, state_space)

        assert p_action[0, 1] == 1 / 4
        assert p_action[0, 3] == 1 / 4
        assert p_action[3, 0] == 1 / 2
        assert p_action[5, 0] == 1 / 2
        assert p_action[14, 3] == 1


class TestExpectedSvfFromPolicy:
    """Tests for function that counts expected empirical feature counts"""

    def test_2_actions(self, n_actions=2) -> None:
        state_space = me.get_state_space(n_actions=n_actions)
        n_states = me.get_n_states(n_actions=n_actions)
        mdp = me.DeterministicTreeMDP(n_actions=n_actions)
        p_transition = me.get_p_transition(n_actions=n_actions, state_space=state_space, mdp=mdp)
        featurizer = me.IdentityFeaturizer(n_actions=n_actions)
        features = me.get_features(featurizer, state_space)
        theta = np.zeros((len(features[0]),)) + 1
        reward = features.dot(theta)
        p_action = me.get_p_action(
            n_states=n_states, n_actions=n_actions, reward=reward, state_space=state_space
        )
        e_svf = me.expected_svf_from_policy(
            n_actions=n_actions, p_transition=p_transition, p_action=p_action
        )

        assert e_svf[0] == 1
        assert e_svf[1] == 0.5
        assert e_svf[2] == 0.5
        assert e_svf[3] == 1

    def test_4_actions(self, n_actions=4) -> None:
        state_space = me.get_state_space(n_actions=n_actions)
        n_states = me.get_n_states(n_actions=n_actions)
        mdp = me.DeterministicTreeMDP(n_actions=n_actions)
        p_transition = me.get_p_transition(n_actions=n_actions, state_space=state_space, mdp=mdp)
        featurizer = me.IdentityFeaturizer(n_actions=n_actions)
        features = me.get_features(featurizer, state_space)
        theta = np.zeros((len(features[0]),)) + 1
        reward = features.dot(theta)
        p_action = me.get_p_action(
            n_states=n_states, n_actions=n_actions, reward=reward, state_space=state_space
        )
        e_svf = me.expected_svf_from_policy(
            n_actions=n_actions, p_transition=p_transition, p_action=p_action
        )

        state1 = state_space.index(tuple([0, 0, 0, 0]))
        state2 = state_space.index(tuple([1, 0, 0, 0]))
        state3 = state_space.index(tuple([1, 1, 1, 0]))

        assert e_svf[state1] == 1
        assert e_svf[state2] == 0.25
        assert e_svf[state3] == 0.25


"""
class TestIRL:
    Tests for the function that performs the descent graident optimization algorithm
    def test_2_actions(self, n_actions = 2):

        trajectories = [
            [(0, 1), (1, 1)], 
            [(0, 0), (0, 1)],
            [(0, 0), (1, 0)],
            [(0, 0), (1, 0), (1, 1)]
        ]

        mdp = me.DeterministicTreeMDP(n_actions)
        featurizer = me.IdentityFeaturizer(n_actions)
        state_space = me.get_state_space(n_actions)
        features = me.get_features(featurizer=featurizer, state_space=state_space)
        n_states = me.get_n_states(n_actions)
        counts = me.expected_empirical_feature_counts(mdp=mdp, featurizer=featurizer, trajectories=trajectories)
        fe = sum(np.array(list(counts.values())))
        optim = O.Sga(lr=0.2)
        

        _, delta_test, _, _, _ = me.irl(n_actions=n_actions, features=features, feature_expectation=fe, optim = optim, 
                 eps=1e-4, mdp=mdp)

        

        for i in range( len(delta_test) - 1 ):
            assert delta_test[i] > delta_test[i+1]

    

    
    def test_3_actions(self, n_actions = 3):

        trajectories = [
                [(0, 0, 0), (0, 1, 0)], 
                [(0, 0, 0), (0, 0, 1)],
                [(0, 0, 0), (1, 0, 0)],
                [(0, 0, 0), (1, 0, 0), (1, 1, 0)]
            ]

        mdp = me.DeterministicTreeMDP(n_actions)
        featurizer = me.IdentityFeaturizer(n_actions)
        state_space = me.get_state_space(n_actions)
        features = me.get_features(featurizer=featurizer, state_space=state_space)
        n_states = me.get_n_states(n_actions)
        counts = me.expected_empirical_feature_counts(mdp=mdp, featurizer=featurizer, trajectories=trajectories)
        fe = sum(np.array(list(counts.values())))
        #optim = O.ExpSga(O.exponential_decay(lr0=0.01))
        optim = O.ExpSga(lr=0.01)
        
        p_initial = np.zeros(n_states)
        p_initial[0] = 1

        _, delta_test, _, _ = me.irl(n_actions,
                    features=features, 
                    feature_expectation=fe, 
                    optim=optim, 
                    p_initial=p_initial, 
                    eps=1e-4, 
                    eps_esvf=1e-5, trajectories=trajectories)

        for i in range( len(delta_test) - 1 ):
            assert delta_test[i] > delta_test[i+1]
"""
