from copy import deepcopy
import numpy as np
from bidict import bidict
from GridGame.BellmanSolver import BellmanSolver
from GridGame.GridWorld import GridWorld
from GridGame.StateTransProb import StateTransProb
from GridGame.RwdTransProb import RwdTransProb
from GridGame.State import State
from GridGame.Action import Action, named_acts, delta_acts, actions, num_actions
from GridGame.Policy import Policy
from GridGame.Pos import Pos
from GridGame.Ppi import Ppi
from GridGame.Rpi import Rpi

import numpy as np

def test_case_1():
    np.random.seed(0)
    N, M = 5, 5

    num_states = N*M
    maps = np.array([
        [' ', ' ', ' ', ' ', ' '],
        [' ', '#', '#', ' ', ' '],
        [' ', ' ', '#', ' ', ' '],
        [' ', '#', 'O', '#', ' '],
        [' ', '#', ' ', ' ', ' '],
    ])
    grid_world = GridWorld.from_maps(maps)
    gamma = 0.9

    policy = np.array([
        ['r', 'r', 'r', 'd', 'd'],
        ['u', 'u', 'r', 'd', 'd'],
        ['u', 'l', 'd', 'r', 'd'],
        ['u', 'r', 's', 'l', 'd'],
        ['u', 'r', 'u', 'l', 'l'],
    ])
    policy = Policy.from_grid(policy, num_actions, named_acts)
    init_v = np.zeros(num_states)
    
    rwd_trans_prob = RwdTransProb.from_grid_world(grid_world)
    state_trans_prob = StateTransProb.from_grid(N, N, num_actions, delta_acts)
    ppi = Ppi(state_trans_prob, policy)
    R_pi = Rpi(policy, rwd_trans_prob)
    print(np.array(R_pi).reshape(N,N))
    solver = BellmanSolver(init_v, np.array(R_pi), gamma, np.array(ppi))
    v = solver.solve()
    print(rwd_trans_prob[4,3])
    print(v.reshape(N,N))
    assert np.allclose(v.reshape(N,N), np.array([
        [3.5, 3.9, 4.3, 4.8, 5.3],
        [3.1, 3.5, 4.8, 5.3, 5.9],
        [2.8, 2.5,10.0, 5.9, 6.6],
        [2.5,10.0,10.0,10.0, 7.3],
        [2.3, 9.0,10.0, 9.0, 8.1],
    ]), atol=1e-1)


def test_case_2():
    np.random.seed(0)
    N = 5

    num_states = N**2
    num_actions = 5
    maps = [
        [' ', ' ', ' ', ' ', ' '],
        [' ', '#', '#', ' ', ' '],
        [' ', ' ', '#', ' ', ' '],
        [' ', '#', 'O', '#', ' '],
        [' ', '#', ' ', ' ', ' '],
    ]
    # generate obstacles and target from maps
    obstacles = []
    target = None
    for i in range(N):
        for j in range(N):
            if maps[i][j] == '#':
                obstacles.append(Pos(i,j))
            
            elif maps[i][j] == 'O':
                target = Pos(i,j)
    grid_world = GridWorld(N, N, obstacles)
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
    m = np.zeros((num_states, num_actions))
    for i in range(policy.shape[0]):
        for j in range(policy.shape[1]):
            m[i*N+j, named_acts[policy[i][j]].id] = 1
    # policy = policy / np.sum(policy, axis=1, keepdims=True)
    policy = Policy(m)
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
    for src in grid_world.neighbors(target) + [target]:
        rwd_trans_prob[src.x*N+src.y, delta_acts[tuple(target-src)].id] = {
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
    print(v.reshape(N,N))
    assert np.allclose(v.reshape(N,N), np.array([
        [3.5, 3.9, 4.3, 4.8, 5.3],
        [3.1, 3.5, 4.8, 5.3, 5.9],
        [2.8, 2.5,10.0, 5.9, 6.6],
        [2.5,10.0,10.0,10.0, 7.3],
        [2.3, 9.0,10.0, 9.0, 8.1],
    ]), atol=1e-1)


def test_case_3():
    np.random.seed(0)
    N = 5

    num_states = N**2
    maps = [
        [' ', ' ', ' ', ' ', ' '],
        [' ', '#', '#', ' ', ' '],
        [' ', ' ', '#', ' ', ' '],
        [' ', '#', 'O', '#', ' '],
        [' ', '#', ' ', ' ', ' '],
    ]
    # generate obstacles and target from maps
    obstacles = []
    target = None
    for i in range(N):
        for j in range(N):
            if maps[i][j] == '#':
                obstacles.append(Pos(i,j))
            
            elif maps[i][j] == 'O':
                target = Pos(i,j)
    grid_world = GridWorld(N, N, obstacles)
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
        ['>', '>', '>', '>', '>'],
        ['>', '>', '>', '>', '>'],
        ['>', '>', '>', '>', '>'],
        ['>', '>', '>', '>', '>'],
        ['>', '>', '>', '>', '>'],
    ])
    m = np.zeros((num_states, num_actions))
    for i in range(policy.shape[0]):
        for j in range(policy.shape[1]):
            m[i*N+j, named_acts[policy[i][j]].id] = 1
    # policy = policy / np.sum(policy, axis=1, keepdims=True)
    policy = Policy(m)
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
    for src in grid_world.neighbors(target) + [target]:
        rwd_trans_prob[src.x*N+src.y, delta_acts[tuple(target-src)].id] = {
            rewards[0]: 0,
            rewards[1]: 0,
            rewards[2]: 1
        }
    # for boundary state
    for i in range(N):
        # top boundary
        rwd_trans_prob[0*N+i, named_acts['^'].id] = {
            rewards[0]: 1,
            rewards[1]: 0,
            rewards[2]: 0
        }
        # bottom boundary
        rwd_trans_prob[(N-1)*N+i, named_acts['v'].id] = {
            rewards[0]: 1,
            rewards[1]: 0,
            rewards[2]: 0
        }
    for i in range(N):
        # left boundary
        rwd_trans_prob[i*N+0, named_acts['<'].id] = {
            rewards[0]: 1,
            rewards[1]: 0,
            rewards[2]: 0
        }
        # right boundary
        rwd_trans_prob[i*N+N-1, named_acts['>'].id] = {
            rewards[0]: 1,
            rewards[1]: 0,
            rewards[2]: 0
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
    print(np.array(R_pi).reshape(N,N))
    solver = BellmanSolver(init_v, np.array(R_pi), gamma, np.array(ppi))
    v = solver.solve()
    print(rwd_trans_prob[0,3])
    print(v.reshape(N,N))
    assert np.allclose(v.reshape(N,N), np.array([
        [ -6.6, -7.3, -8.1, -9.0,-10.0],
        [ -8.5, -8.3, -8.1, -9.0,-10.0],
        [ -7.5, -8.3, -8.1, -9.0,-10.0],
        [ -7.5, -7.2, -9.1, -9.0,-10.0],
        [ -7.6, -7.3, -8.1, -9.0,-10.0],
    ]), atol=1e-1)


def test_case_4():
    np.random.seed(0)
    N, M = 5, 5

    num_states = N*M
    maps = np.array([
        [' ', ' ', ' ', ' ', ' '],
        [' ', '#', '#', ' ', ' '],
        [' ', ' ', '#', ' ', ' '],
        [' ', '#', 'O', '#', ' '],
        [' ', '#', ' ', ' ', ' '],
    ])
    grid_world = GridWorld.from_maps(maps)
    gamma = 0.9

    policy = np.array([
        ['>', '<', '<', '^', '^'],
        ['v', 'o', '>', 'v', '>'],
        ['<', '>', 'v', '<', 'o'],
        ['o', 'v', '^', '^', '>'],
        ['o', '>', 'o', '>', 'o'],
    ])
    policy = Policy.from_grid(policy, num_actions, named_acts)
    init_v = np.zeros(num_states)
    
    rwd_trans_prob = RwdTransProb.from_grid_world(grid_world)
    state_trans_prob = StateTransProb.from_grid(N, N, num_actions, delta_acts)
    ppi = Ppi(state_trans_prob, policy)
    R_pi = Rpi(policy, rwd_trans_prob)
    print(np.array(R_pi).reshape(N,N))
    solver = BellmanSolver(init_v, np.array(R_pi), gamma, np.array(ppi))
    v = solver.solve()
    print(rwd_trans_prob[4,3])
    print(v.reshape(N,N))
    assert np.allclose(v.reshape(N,N), np.array([
        [  0.0,  0.0,  0.0,-10.0,-10.0],
        [ -9.0,-10.0, -0.4, -0.5,-10.0],
        [-10.0, -0.5,  0.5, -0.5,  0.0],
        [  0.0, -1.0, -0.5, -0.5,-10.0],
        [  0.0,  0.0,  0.0,  0.0,  0.0],
    ]), atol=1e-1)










