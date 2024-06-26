from typing import Optional, Union, Dict, List
import numpy as np
from copy import deepcopy
from State import State
from Action import Action
from GridWorld import GridWorld

class RwdTransProb:
    '''
    Shape: (num_states, num_actions)
    '''
    def __init__(self, m: np.ndarray):
        # make sure m is a 2D array
        if len(m.shape)!= 2:
            raise ValueError("m should be a 2D array")
        # make sure the sum of all values in each elements is 1
        if np.any(np.vectorize(lambda x: sum(x.values()) != 1)(m)):
            raise ValueError("The sum of all values in each element should be 1")
        self.m = m
    

    def __call__(self, src: State, action: Action, reward: Optional[float] = None) -> Union[Dict[float, float], float]:
        if reward is None:
            return self.m[src.id, action.id]
        else:
            return self.m[src.id, action.id][reward]
    

    def __getitem__(self, key):
        return self.m[key]


    @classmethod
    def from_grid_world(cls, grid_world: GridWorld, rewards: Optional[List[float]] = None) -> 'RwdTransProb':
        N, M = grid_world.n, grid_world.m
        num_states = N*M
        num_actions = 5
        actions = [Action(i) for i in range(num_actions)]
        delta_acts = {
            (-1,0): actions[0], (1,0): actions[1],
            (0,-1): actions[2], (0,1): actions[3],
            (0,0): actions[4]
        }
        named_acts = {
            "u": actions[0], "d": actions[1],
            "l": actions[2], "r": actions[3],
            "s": actions[4],
            "^": actions[0], "v": actions[1],
            "<": actions[2], ">": actions[3],
            "o": actions[4]
        }
        if rewards is None:
            rewards = [-1, 0, 1]
        default_rewards = {rewards[0]: 0, rewards[1]: 1, rewards[2]: 0}
        rwd_trans_prob = np.array([[deepcopy(default_rewards) for _ in range(num_actions)] for _ in range(num_states)])
        for obstacle in grid_world.obstacles:
            for src in grid_world.neighbors(obstacle)+[obstacle]:
                rwd_trans_prob[src.x*M+src.y, delta_acts[tuple(obstacle-src)].id] = {
                    rewards[0]: 1,
                    rewards[1]: 0,
                    rewards[2]: 0
                }
        for src in grid_world.neighbors(grid_world.target) + [grid_world.target]:
            rwd_trans_prob[src.x*M+src.y, delta_acts[tuple(grid_world.target-src)].id] = {
                rewards[0]: 0,
                rewards[1]: 0,
                rewards[2]: 1
            }
        # for boundary state
        for i in range(M):
            # top boundary
            rwd_trans_prob[0*M+i, named_acts['^'].id] = {
                rewards[0]: 1,
                rewards[1]: 0,
                rewards[2]: 0
            }
            # bottom boundary
            rwd_trans_prob[(N-1)*M+i, named_acts['v'].id] = {
                rewards[0]: 1,
                rewards[1]: 0,
                rewards[2]: 0
            }
        for i in range(N):
            # left boundary
            rwd_trans_prob[i*M+0, named_acts['<'].id] = {
                rewards[0]: 1,
                rewards[1]: 0,
                rewards[2]: 0
            }
            # right boundary
            rwd_trans_prob[i*M+M-1, named_acts['>'].id] = {
                rewards[0]: 1,
                rewards[1]: 0,
                rewards[2]: 0
            }
        return cls(rwd_trans_prob)




















