import numpy as np
from State import State
from Action import Action

class StateTransProb:
    '''
    This class represents the transition probability of a state.
    '''
    def __init__(self, m: np.ndarray) -> None:
        # make sure the m has 3 dimensions
        if len(m.shape)!= 3:
            raise ValueError("The transition probability matrix should have 3 dimensions.")
        # Constraint
        if not np.allclose(np.sum(m, axis=2), 1):
            raise ValueError("The sum of transition probability matrix should be 1 for each state-action pair.")
        self.m = m
    

    def __call__(self, dest: State, src: State, action: Action) -> float:
        return self.m[src, action, dest]
