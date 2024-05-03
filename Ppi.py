from typing import List, Dict, Tuple
import numpy as np
from GridGame.State import State
from GridGame.StateTransProb import StateTransProb
from GridGame.Policy import Policy


class Ppi:
    '''
    Shape: (num_states, num_states)
    '''
    def __init__(self, state_trans_prob: StateTransProb, policy: Policy):
        self.state_trans_prob = state_trans_prob
        self.policy = policy
        self.states = list(map(lambda x: State(x), range(self.state_trans_prob.m.shape[0])))
        if not np.allclose(np.sum(self.__array__(), axis=1), 1.0):
            raise ValueError("The sum of transition probabilities for each state is not equal to 1.")

    def __call__(self, src: State, dest: State) -> float:
        if self.state_trans_prob.m.shape[1] != self.policy.m.shape[1]:
            m, n= self.state_trans_prob.m.shape[1], self.policy.m.shape[1]
            raise ValueError(f"The number of actions in state transition probability matrix ({m}) and policy matrix ({n}) are not equal.")
        return np.dot(
            self.state_trans_prob[src.id, :, dest.id], 
            self.policy[src.id]
        )

    def __array__(self) -> np.ndarray:
        shape = (self.state_trans_prob.m.shape[0], self.state_trans_prob.m.shape[0])
        arr = np.zeros(shape)
        for src in self.states:
            for dest in self.states:
                arr[src.id, dest.id] = self.__call__(src, dest)
        return arr

