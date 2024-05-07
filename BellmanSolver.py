import numpy as np

from GridGame.GridWorld import GridWorld
from GridGame.StateTransProb import StateTransProb
from GridGame.RwdTransProb import RwdTransProb
from GridGame.Action import Action, named_acts, delta_acts, actions, num_actions
from GridGame.Policy import Policy
from GridGame.Pos import Pos
from GridGame.Ppi import Ppi
from GridGame.Rpi import Rpi


class BellmanSolver:
    
    @classmethod
    def from_policy(cls, rwd_trans_prob: RwdTransProb, state_trans_prob: StateTransProb, policy: Policy, gamma) -> "BellmanSolver":
        np.random.seed(0)
        
        ppi = Ppi(state_trans_prob, policy)
        R_pi = Rpi(policy, rwd_trans_prob)
        return BellmanSolver(np.array(R_pi), gamma, np.array(ppi))
    
    def __init__(self, r, gamma, P):
        self.r = r
        self.gamma = gamma
        self.P = P
    

    def solve(self, epsilon=1e-4, max_iterations=1000) -> np.ndarray:
        last = np.zeros(self.P.shape[0])
        for _ in range(max_iterations):
            v_k = self.r + self.gamma * np.dot(self.P, last)
            if np.max(np.abs(v_k - last)) < epsilon:
                break
            last = v_k.copy()
        else:
            raise ValueError("Failed to converge")
        return v_k
    

