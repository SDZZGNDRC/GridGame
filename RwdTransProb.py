import numpy as np
from State import State
from Action import Action

class RwdTransProb:
    def __init__(self, m: np.ndarray):
        # make sure m is a 2D array
        if len(m.shape)!= 2:
            raise ValueError("m should be a 2D array")
        # make sure the sum of all values in each elements is 1
        if not np.allclose(list(map(lambda x: sum(x.item().values()), np.nditer(m, flags=['refs_ok']))), 1):
            raise ValueError("The sum of all values in each element should be 1")
        self.m = m
    

    def __call__(self, src: State, action: Action, reward: float) -> float:
        return self.m[src, action][reward]

























