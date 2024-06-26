from typing import Dict
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
    

    @classmethod
    def from_grid(cls, grid: np.ndarray, num_actions: int, named_acts: Dict[str, Action]) -> 'Policy':
        N, M = grid.shape
        m = np.zeros((grid.size, num_actions))
        for i in range(N):
            for j in range(M):
                m[i*M+j, named_acts[grid[i][j]].id] = 1
        return cls(m)

    @classmethod
    def random(cls, num_states: int, num_actions: int) -> 'Policy':
        m = np.random.rand(num_states, num_actions)
        return cls(m / m.sum(axis=1, keepdims=True))  # normalize the rows to sum up to 1

