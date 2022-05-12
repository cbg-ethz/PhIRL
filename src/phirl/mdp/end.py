from typing import cast, Tuple, Union

import numpy as np

import phirl.mdp.common as common
import phirl.mdp.interfaces as interfaces
import phirl.mdp.simple as simple

# We allow an additional action leading to the unique end state
END_ACTION = "END ACTION"
END_STATE = "END STATE"

# Hence, we need to extend the definition of the state
State = Union[simple.State, type(END_STATE)]
Action = Union[simple.Action, type(END_ACTION)]


class EndParams(interfaces.IMDPParams):
    def __init__(self, n_ordinary_actions: int) -> None:
        self.simple = simple.SimpleParams(n_actions=n_ordinary_actions)

    @property
    def states(self) -> Tuple[State]:
        return cast(Tuple[State], tuple(list(self.simple.states) + [END_STATE]))

    @property
    def actions(self) -> Tuple[Action]:
        return cast(Tuple[Action], tuple(list(self.simple.actions) + [END_ACTION]))

    @property
    def n_states(self) -> int:
        return self.simple.n_states + 1

    @property
    def n_actions(self) -> int:
        return self.simple.n_actions + 1


class EndIdentityFeaturizer(interfaces.IFeaturizer):
    def __init__(self, params: EndParams) -> None:
        self._params = params
        self._featurizer = simple.IdentityFeaturizer(params.simple)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._featurizer.shape

    def transform(self, state: State) -> interfaces.Feature:
        if state == END_STATE:
            return np.zeros(self.shape)
        else:
            state = cast(simple.State, state)
            return self._featurizer.transform(state)


def add_end_action_and_state(
    trajectory: common.Trajectory,
) -> common.Trajectory:
    """Creates a new trajectory, differing from `trajectory` by the end action and the end state."""
    return common.Trajectory(
        states=list(trajectory.states) + [END_STATE],
        actions=list(trajectory.actions) + [END_ACTION],
    )
