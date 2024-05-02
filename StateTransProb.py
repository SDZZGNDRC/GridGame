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
        # make sure the sum over the last two dimensions is 1
        if not np.allclose(np.sum(m, axis=(1, 2)), 1):
            raise ValueError("The sum over the last two dimensions of m should be 1")
        self.m = m
    

    def __call__(self, dest: State, src: State, action: Action) -> float:
        return self.m[src, action, dest]
