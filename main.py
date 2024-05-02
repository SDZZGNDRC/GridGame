from copy import deepcopy
import numpy as np
from bidict import bidict
from BellmanSolver import BellmanSolver
from GridWorld import GridWorld
from StateTransProb import StateTransProb
from RwdTransProb import RwdTransProb
from State import State
from Action import Action
from Policy import Policy
from Pos import Pos
from Ppi import Ppi
from Rpi import Rpi

import numpy as np

def main():
    np.random.seed(0)
    N = 5

    num_states = N**2
    num_actions = 5
    obstacles = Pos.from_list([
        (1,1), (1,2),
        (2,2),
        (3,1), (3,3),
        (4,1)
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
    
    policy = np.random.rand(num_states, num_actions)
    # normalize policy
    policy = policy / np.sum(policy, axis=1, keepdims=True)
    policy = Policy(policy)
    init_v = np.zeros(num_states)
    default_rewards = {rewards[0]: 0, rewards[1]: 1, rewards[2]: 0}
    rwd_trans_prob = np.array([[deepcopy(default_rewards) for _ in range(num_actions)] for _ in range(num_states)])
    for obstacle in grid_world.obstacles:
        for src in grid_world.neighbors(obstacle):
            rwd_trans_prob[src.x*N+src.y, delta_acts[tuple(obstacle-src)].id] = {
                rewards[0]: 1,
                rewards[1]: 0,
                rewards[2]: 0
            }
    target_pos = Pos(target[0], target[1])
    for src in grid_world.neighbors(target_pos):
        rwd_trans_prob[src.x*N+src.y, delta_acts[tuple(target_pos-src)].id] = {
            rewards[0]: 0,
            rewards[1]: 0,
            rewards[2]: 1
        }
    rwd_trans_prob = RwdTransProb(rwd_trans_prob)
    state_trans_prob = np.zeros((num_states, num_actions, num_states))
    for x in range(N):
        for y in range(N):
            state_trans_prob[x*N+y, delta_acts[(0,0)].id, x*N+y] = 1
            state_trans_prob[x*N+y, delta_acts[(-1,0)].id, max(x-1,0)*N+y] = 1
            state_trans_prob[x*N+y, delta_acts[(1,0)].id, min(x+1,N-1)*N+y] = 1
            state_trans_prob[x*N+y, delta_acts[(0,-1)].id, x*N+max(y-1,0)] = 1
            state_trans_prob[x*N+y, delta_acts[(0,1)].id, x*N+min(y+1,N-1)] = 1
    state_trans_prob = StateTransProb(state_trans_prob)
    ppi = Ppi(state_trans_prob, policy)
    R_pi = Rpi(policy, rwd_trans_prob)
    solver = BellmanSolver(init_v, np.array(R_pi), gamma, np.array(ppi))
    v = solver.solve()
    print(v)
    print(np.max(v))


if __name__ == '__main__':
    main()



