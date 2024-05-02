import numpy as np
from typing import Dict
from State import State
from Action import Action
from Policy import Policy
from RwdTransProb import RwdTransProb


class Rpi:
    
    @staticmethod
    def __convert(rewards: Dict[float, float]) -> float:
        return sum([r * p for r, p in rewards.items()])
    
    def __init__(self, policy: Policy, rwd_trans_prob: RwdTransProb) -> None:
        self.policy = policy
        self.rwd_trans_prob = rwd_trans_prob
        self.states = list(map(lambda x: State(x), range(self.rwd_trans_prob.m.shape[0])))
        self.actions = list(map(lambda x: Action(x), range(self.rwd_trans_prob.m.shape[1])))
        self.__data = np.sum(np.vectorize(self.__convert)(self.rwd_trans_prob.m), axis=1)
    

    def __call__(self, src: State) -> float:
        return self.__data[src.id]
    

    def __array__(self) -> np.ndarray:
        return self.__data.copy()
































