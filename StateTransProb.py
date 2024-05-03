from typing import Dict, Tuple
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
    

    @classmethod
    def from_grid(cls, N: int, M: int, num_actions: int, delta_acts: Dict[Tuple[int,int], Action]) -> 'StateTransProb':
        num_states = N*M
        
        state_trans_prob = np.zeros((num_states, num_actions, num_states))
        for x in range(N):
            for y in range(N):
                state_trans_prob[x*N+y, delta_acts[(0,0)].id, x*N+y] = 1
                state_trans_prob[x*N+y, delta_acts[(-1,0)].id, max(x-1,0)*N+y] = 1
                state_trans_prob[x*N+y, delta_acts[(1,0)].id, min(x+1,N-1)*N+y] = 1
                state_trans_prob[x*N+y, delta_acts[(0,-1)].id, x*N+max(y-1,0)] = 1
                state_trans_prob[x*N+y, delta_acts[(0,1)].id, x*N+min(y+1,N-1)] = 1
        return cls(state_trans_prob)
