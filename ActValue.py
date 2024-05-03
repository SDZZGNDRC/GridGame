from typing import List, Dict, Tuple
import numpy as np

from GridGame.State import State
from GridGame.Action import Action
from GridGame.RwdTransProb import RwdTransProb
from GridGame.StateTransProb import StateTransProb


class ActValue:
    
    @classmethod
    def from_v_pi(cls, v_pi: np.ndarray, gamma: float, rwd_trans_prob: RwdTransProb, state_trans_prob: StateTransProb) -> 'ActValue':
        raise NotImplementedError()

    def __init__(self, m: np.ndarray):
        self.m = m

    def __call__(self, s: State, a: Action) -> float:
        pass

    def __array__(self) -> np.ndarray:
        return self.m












