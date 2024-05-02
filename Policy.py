import numpy as np

from State import State
from Action import Action


class Policy:
    '''
    Shape of m: (num_states, num_actions)
    '''
    def __init__(self, m: np.ndarray) -> None:
        # make sure m is a valid transition matrix
        if len(m.shape) != 2:
            raise ValueError("m must be a 2D array")
        # the sum of each row should be 1
        if not np.allclose(np.sum(m, axis=1), 1):
            raise ValueError("m must be a valid transition matrix")
        self.m = m
        

    def __getitem__(self, key):
        return self.m[key]
    

    def __call__(self, src: State, act: Action) -> float:
        return self.m[src.id, act.id]

    def __str__(self) -> str:
        return f"Pi({self.m})"
    

    def __repr__(self) -> str:
        return f"Pi({self.m})"
    




