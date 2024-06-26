import numpy as np
from typing import Literal, Tuple

from BellmanSolver import BellmanSolver
from StateTransProb import StateTransProb
from RwdTransProb import RwdTransProb
from Policy import Policy
from ActValue import ActValue

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

    def solve(self, epsilon=1e-4, max_iterations=1000, method: Literal["policy_iter", "value_iter", "truncated_policy_iter"] = "value_iter") -> np.ndarray:
        if method == "policy_iter":
            return self.policy_iteration(epsilon, max_iterations)
        elif method == "value_iter":
            return self.value_iteration(epsilon, max_iterations)
        elif method == "truncated_policy_iter":
            return self.truncated_policy_iteration(epsilon, max_iterations)
        else:
            raise ValueError("Invalid method.")


    def value_iteration(self, epsilon=1e-4, max_iterations=1000) -> Tuple[np.ndarray, np.ndarray]:
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


    def policy_iteration(self, epsilon=1e-4, max_iterations=1000) -> Tuple[np.ndarray, np.ndarray]:
        policy_k = Policy.random(self.num_states, self.num_actions)
        for _ in range(max_iterations):
            # policy evaluation
            v_pi_k = BellmanSolver.from_policy(self.rwd_trans_prob, self.state_trans_prob, policy_k, self.gamma).solve(epsilon, max_iterations)
            
            # policy improvement
            q_pi_k = ActValue.from_v_pi(v_pi_k, self.gamma, self.rwd_trans_prob, self.state_trans_prob)
            a_pi_k_star = np.argmax(q_pi_k, axis=1)
            policy_k_new_m = np.zeros((self.num_states, self.num_actions))
            for s in range(self.num_states):
                policy_k_new_m[s, a_pi_k_star[s]] = 1
            policy_k_new = Policy(policy_k_new_m)
            if np.max(np.abs(policy_k_new.m - policy_k.m)) < epsilon:
                break
            policy_k = policy_k_new
        else:
            raise ValueError("Failed to converge in the given number of iterations.")
        return v_pi_k, policy_k.m


    def truncated_policy_iteration(self, epsilon=1e-4, max_iterations=1000) -> Tuple[np.ndarray, np.ndarray]:
        policy_k = Policy.random(self.num_states, self.num_actions)
        for _ in range(max_iterations):
            # policy evaluation
            v_pi_k = BellmanSolver.from_policy(self.rwd_trans_prob, self.state_trans_prob, policy_k, self.gamma).solve(epsilon, 50, truncated=True)
            
            # policy improvement
            q_pi_k = ActValue.from_v_pi(v_pi_k, self.gamma, self.rwd_trans_prob, self.state_trans_prob)
            a_pi_k_star = np.argmax(q_pi_k, axis=1)
            policy_k_new_m = np.zeros((self.num_states, self.num_actions))
            for s in range(self.num_states):
                policy_k_new_m[s, a_pi_k_star[s]] = 1
            policy_k_new = Policy(policy_k_new_m)
            if np.max(np.abs(policy_k_new.m - policy_k.m)) < epsilon:
                break
            policy_k = policy_k_new
        else:
            raise ValueError("Failed to converge in the given number of iterations.")
        return v_pi_k, policy_k.m



