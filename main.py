from copy import deepcopy
import numpy as np
from GridGame.BellmanSolver import BellmanSolver
from GridGame.GridWorld import GridWorld
from GridGame.StateTransProb import StateTransProb
from GridGame.RwdTransProb import RwdTransProb
from GridGame.State import State
from GridGame.Action import Action
from GridGame.Policy import Policy
from GridGame.Pos import Pos
from GridGame.Ppi import Ppi
from GridGame.Rpi import Rpi

import numpy as np

def main():
    np.random.seed(0)
    N = 5

    num_states = N**2
    num_actions = 5
    maps = np.array([
        [' ', ' ', ' ', ' ', ' '],
        [' ', '#', '#', ' ', ' '],
        [' ', ' ', '#', ' ', ' '],
        [' ', '#', 'O', '#', ' '],
        [' ', '#', ' ', ' ', ' '],
    ])
    grid_world = GridWorld.from_maps(maps)
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
    rewards = [-1, 0, 1]
    gamma = 0.9
    
    # policy = np.random.rand(num_states, num_actions)
    # normalize policy
    policy = np.array([
        ['>', '>', '>', 'v', 'v'],
        ['^', '^', '>', 'v', 'v'],
        ['^', '<', 'v', '>', 'v'],
        ['^', '>', 'o', '<', 'v'],
        ['^', '>', '^', '<', '<'],
    ])
    policy = Policy.from_grid(policy, num_actions, named_acts)
    init_v = np.zeros(num_states)

    rwd_trans_prob = RwdTransProb.from_grid_world(grid_world)
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
    print("R_pi:\n", np.array(R_pi).reshape(N, N))
    print(v.reshape(N, N))
    print(np.max(v))


if __name__ == '__main__':
    np.set_printoptions(precision=1)
    main()



