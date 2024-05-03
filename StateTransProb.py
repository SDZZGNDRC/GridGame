import numpy as np
from GridGame.State import State
from GridGame.Action import Action

class StateTransProb:
    '''
    This class represents the transition probability of a state.
    Shape of the matrix: (num_states, num_actions, num_states)
    '''
    def __init__(self, m: np.ndarray) -> None:
        # make sure the m has 3 dimensions
        if len(m.shape)!= 3:
            raise ValueError("The transition probability matrix should have 3 dimensions.")
        # Constraint
        if m.shape[0]!= m.shape[2]:
            raise ValueError("The first and third dimension of the transition probability matrix should be the same.")
        if not np.allclose(np.sum(m, axis=2), 1):
            print(np.sum(m, axis=2))
            # print(m[0, 0, :].shape)
            raise ValueError("The sum of transition probability matrix should be 1 for each state-action pair.")
        self.m = m
    

    def __call__(self, src: State, action: Action, dest: State) -> float:
        return self.m[src.id, action.id, dest.id]
    

    def __getitem__(self, key):
        return self.m[key]
