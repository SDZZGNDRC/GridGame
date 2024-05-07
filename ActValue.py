from typing import List, Dict, Tuple
import numpy as np

from GridGame.State import State
from GridGame.Action import Action
from GridGame.RwdTransProb import RwdTransProb
from GridGame.StateTransProb import StateTransProb


class ActValue:
    
    @classmethod
    def from_v_pi(cls, v_pi: np.ndarray, gamma: float, rwd_trans_prob: RwdTransProb, state_trans_prob: StateTransProb) -> 'ActValue':
        if rwd_trans_prob.m.shape != state_trans_prob.m.shape[:2]:
            raise ValueError("rwd_trans_prob and state_trans_prob should have the same number of states and actions")
        num_states, num_actions = rwd_trans_prob.m.shape
        
        m = np.zeros((num_states, num_actions))
        for s in range(num_states):
            for a in range(num_actions):
                r_s_a = rwd_trans_prob[s, a]
                p1 = sum(map(lambda r: r*rwd_trans_prob.m[s, a][r], r_s_a.keys()))
                p2 = gamma * sum(map(lambda dst: state_trans_prob.m[s, a, dst]*v_pi[dst], range(num_states)))
                m[s, a] = p1 + p2
        return cls(m)

    def __init__(self, m: np.ndarray):
        # make sure the m has 2 dimensions
        if len(m.shape)!= 2:
            raise ValueError("m should have 2 dimensions")
        self.m = m

    def __call__(self, s: State, a: Action) -> float:
        return self.m[s.idx, a.idx]

    def __array__(self) -> np.ndarray:
        return self.m












