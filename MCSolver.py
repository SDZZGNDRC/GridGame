import numpy as np
from typing import Literal, Tuple

from MCInterface import MCInterface
from StateTransProb import StateTransProb
from RwdTransProb import RwdTransProb
from Policy import Policy
from Action import named_acts, actions
from GridWorld import GridWorld


class MCSolver:
    def __init__(self, gw: GridWorld, mci: MCInterface):
        self.grid_world = gw
        self.mci = mci
        self.num_states = len(gw.states)
        self.num_actions = len(actions)

    def solve(self, epsilon=1e-4, max_iterations=1000, episode: int = 1000, method: Literal["MC Basic", "MC Exploring Starts"] = "MC Basic"):
        if method == "MC Basic":
            return self.mc_basic(epsilon, max_iterations, episode)
        else:
            raise ValueError("Invalid method")

    def mc_basic(self, epsilon=1e-4, max_iterations=1000, episode: int = 1000):
        policy_k = Policy.random(self.num_states, self.num_actions)
        for _ in range(max_iterations):
            q_pi = np.zeros((self.num_states, self.num_actions))

            for state in self.grid_world.states:
                for action in actions:
                    # collect an episode
                    eps = self.mci.episode(policy_k, state, action, length=episode) # [(state, action, reward), ...]
                    q_pi[state.id, action.id] = np.mean(map(lambda x: x[2], eps))
            # Policy improvement
            a_pi_k_star = np.argmax(q_pi, axis=1)
            policy_k_new_m = np.zeros((self.num_states, self.num_actions))
            for s in range(self.num_states):
                policy_k_new_m[s, a_pi_k_star[s]] = 1
            policy_k_new = Policy(policy_k_new_m)
            if np.max(np.abs(policy_k_new.m - policy_k.m)) < epsilon:
                break
            policy_k = policy_k_new
        else:
            raise ValueError("Failed to converge in the given number of iteration.")
        return policy_k












