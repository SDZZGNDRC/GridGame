import numpy as np
from bidict import bidict
from ..BellmanSolver import BellmanSolver
from ..GridWorld import GridWorld
from ..StateTransProb import StateTransProb
from ..RwdTransProb import RwdTransProb
from ..State import State
from ..Action import Action
from ..Policy import Policy
from ..Pos import Pos
from ..Ppi import Ppi
from ..RPi import RPi

import numpy as np

def test_case_1():
    np.random.seed(0)
    N = 5

    num_states = N**2
    num_actions = 5
    obstacles = Pos.from_list([
        (2,2), (2,3),
        (3,3),
        (4,2), (4,4),
        (5,2)
    ])
    target = (4,3)
    grid_world = GridWorld(N, N, obstacles)
    actions = [Action(i) for i in range(num_actions)]
    delta_acts = {
        (-1,0): actions[0], (1,0): actions[1],
        (0,-1): actions[2], (0,1): actions[3],
        (0,0): actions[4]
    }
    rewards = [-1, 0, 1]
    gamma = 0.9
    
    policy = Policy(np.random.rand(num_states, num_actions))
    init_v = np.zeros(N)
    rwd_trans_prob = np.zeros((num_states, num_actions, num_states))
    for obstacle in grid_world.obstacles:
        for src in grid_world.neighbors(obstacle):
            rwd_trans_prob[src.x*N+src.y, delta_acts[tuple(obstacle-src)].id, obstacle.x*N+obstacle.y] = rewards[0]
    target_pos = Pos(target[0], target[1])
    for src in grid_world.neighbors(target_pos):
        rwd_trans_prob[src.x*N+src.y, delta_acts[tuple(target_pos-src)].id, target_pos.x*N+target_pos.y] = rewards[2]
    rwd_trans_prob = RwdTransProb(rwd_trans_prob)
    state_trans_prob = np.zeros((num_states, num_actions, num_states))
    for x in range(N):
        for y in range(N):
            src = Pos(x, y)
            for neighbor in grid_world.neighbors(src):
                state_trans_prob[src.x*N+src.y, delta_acts[tuple(neighbor-src)].id, neighbor.x*N+neighbor.y] = 1
    state_trans_prob = StateTransProb(state_trans_prob)
    ppi = Ppi(state_trans_prob, rwd_trans_prob)
    R_pi = RPi(policy, rwd_trans_prob)
    solver = BellmanSolver(init_v, np.array(R_pi), gamma, np.array(ppi))
    v = solver.solve()
    print(v)


if __name__ == '__main__':
    test_case_1()

















