from typing import Tuple, Optional, Literal, List, Union

from State import State
from Action import Action
from StateTransProb import StateTransProb
from RwdTransProb import RwdTransProb
from Policy import Policy

class MCInterface:
    '''
    As an interface between the Monte Carlo solver and the grid world
    '''
    def __init__(self, rwd: RwdTransProb, stat: StateTransProb) -> None:
        self.rwd = rwd
        self.stat = stat

    def sample(self, state: Union[int, State], action: Union[int, Action]) -> Tuple[float, State]:
        """
        Sample a reward and next state from the environment given a state and action.
        """
        return (self.rwd.sample(state, action), self.stat.sample(state, action))

    def episode(self, policy: Policy, start: Union[int, State], action: Union[int, Action], length: int = 1000) -> List[Tuple[State, Action, float]]:
        """
        generate an episode with given length
        """
        eps = []
        s = start
        a = action
        for _ in range(length):
            r, nxt_s = self.sample(s, a)
            eps.append((s, a, r))
            s = nxt_s
            a = policy.sample(s)
        
        return eps





