import numpy as np


class BellmanSolver:
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
    

