import numpy as np
from typing import Literal

from GridGame.StateTransProb import StateTransProb
from GridGame.RwdTransProb import RwdTransProb
from GridGame.Policy import Policy
from GridGame.ActValue import ActValue

class BOESolver:
    def __init__(self, gamma: float, state_trans_prob: StateTransProb, rwd_trans_prob: RwdTransProb):
        # make sure the numbers of states and actions are the same
        if state_trans_prob.m.shape[:2] != rwd_trans_prob.m.shape:
            raise ValueError("The numbers of states and actions are not the same.")
        self.state_trans_prob = state_trans_prob
        self.rwd_trans_prob = rwd_trans_prob
        self.gamma = gamma
        self.num_states = state_trans_prob.m.shape[0]
        self.num_actions = state_trans_prob.m.shape[1]

    def solve(self, epsilon=1e-4, max_iterations=1000, method: Literal["policy_iter", "value_iter"] = "value_iter") -> np.ndarray:
        if method == "policy_iter":
            raise NotImplementedError("Policy iteration is not implemented yet.")
        elif method == "value_iter":
            return self.value_iteration(epsilon, max_iterations)
        else:
            raise ValueError("Invalid method.")


    def value_iteration(self, epsilon=1e-4, max_iterations=1000) -> np.ndarray:
        v_pi_k = np.zeros(self.num_states)
        for _ in range(max_iterations):
            q_k = ActValue.from_v_pi(
                v_pi_k, self.gamma, 
                self.rwd_trans_prob, self.state_trans_prob
            )
            a_k_star = np.argmax(q_k, axis=1)
            pi = np.zeros((self.num_states, self.num_actions))
            for s in range(self.num_states):
                pi[s, a_k_star[s]] = 1
            v_pi_k_new = np.max(q_k, axis=1)
            if np.max(np.abs(v_pi_k_new - v_pi_k)) < epsilon:
                break
            v_pi_k = v_pi_k_new
        else:
            raise ValueError("Failed to converge in the given number of iterations.")
        return v_pi_k, pi









