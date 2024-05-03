import numpy as np
from GridGame.State import State
from GridGame.Action import Action

class RwdTransProb:
    '''
    Shape: (num_states, num_actions)
    '''
    def __init__(self, m: np.ndarray):
        # make sure m is a 2D array
        if len(m.shape)!= 2:
            raise ValueError("m should be a 2D array")
        # make sure the sum of all values in each elements is 1
        if np.any(np.vectorize(lambda x: sum(x.values()) != 1)(m)):
            raise ValueError("The sum of all values in each element should be 1")
        self.m = m
    

    def __call__(self, src: State, action: Action, reward: float) -> float:
        return self.m[src.id, action.id][reward]
    

    def __getitem__(self, key):
        return self.m[key]























