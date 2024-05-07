import numpy as np
import inspect
from pathlib import Path
from GridGame.BOESolver import BOESolver
from GridGame.GridWorld import GridWorld
from GridGame.StateTransProb import StateTransProb
from GridGame.RwdTransProb import RwdTransProb
from GridGame.Action import delta_acts, num_actions


def test_case_1():
    np.random.seed(0)
    N, M = 5, 5

    maps = np.array([
        [' ', ' ', ' ', ' ', ' '],
        [' ', '#', '#', ' ', ' '],
        [' ', ' ', '#', ' ', ' '],
        [' ', '#', 'O', '#', ' '],
        [' ', '#', ' ', ' ', ' '],
    ])
    grid_world = GridWorld.from_maps(maps)
    gamma = 0.9
    
    rwd_trans_prob = RwdTransProb.from_grid_world(grid_world)
    state_trans_prob = StateTransProb.from_grid(N, M, num_actions, delta_acts)
    solver = BOESolver(gamma, state_trans_prob, rwd_trans_prob)
    v_star, pi_star = solver.solve()
    print(v_star.reshape(N,M))
    grid_world.draw_policy(pi_star, out_file=Path(f'./test_BOESolver_{inspect.currentframe().f_code.co_name}_policy.png'))
    grid_world.draw_v_pi(v_star, out_file=Path(f'./test_BOESolver_{inspect.currentframe().f_code.co_name}_v_pi.png'))
    assert np.allclose(v_star.reshape(N,M), np.array([
        [5.8, 5.6, 6.2, 6.5, 5.8],
        [6.5, 7.2, 8.0, 7.2, 6.5],
        [7.2, 8.0,10.0, 8.0, 7.2],
        [8.0,10.0,10.0,10.0, 8.0],
        [7.2, 9.0,10.0, 9.0, 8.1],
    ]), atol=1e-1)


def test_case_2():
    np.random.seed(0)
    N, M = 5, 5

    maps = np.array([
        [' ', ' ', ' ', ' ', ' '],
        [' ', '#', '#', ' ', ' '],
        [' ', ' ', '#', ' ', ' '],
        [' ', '#', 'O', '#', ' '],
        [' ', '#', ' ', ' ', ' '],
    ])
    grid_world = GridWorld.from_maps(maps)
    gamma = 0.5
    
    rwd_trans_prob = RwdTransProb.from_grid_world(grid_world)
    state_trans_prob = StateTransProb.from_grid(N, M, num_actions, delta_acts)
    solver = BOESolver(gamma, state_trans_prob, rwd_trans_prob)
    v_star, pi_star = solver.solve()
    print(v_star.reshape(N,M))
    grid_world.draw_policy(pi_star, out_file=Path(f'./test_BOESolver_{inspect.currentframe().f_code.co_name}_policy.png'))
    grid_world.draw_v_pi(v_star, out_file=Path(f'./test_BOESolver_{inspect.currentframe().f_code.co_name}_v_pi.png'))
    assert np.allclose(v_star.reshape(N,M), np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.1],
        [0.0, 0.0, 2.0, 0.1, 0.1],
        [0.0, 2.0, 2.0, 2.0, 0.2],
        [0.0, 1.0, 2.0, 1.0, 0.5],
    ]), atol=1e-1)


def test_case_3():
    np.random.seed(0)
    N, M = 5, 5

    maps = np.array([
        [' ', ' ', ' ', ' ', ' '],
        [' ', '#', '#', ' ', ' '],
        [' ', ' ', '#', ' ', ' '],
        [' ', '#', 'O', '#', ' '],
        [' ', '#', ' ', ' ', ' '],
    ])
    grid_world = GridWorld.from_maps(maps)
    gamma = 0
    
    rwd_trans_prob = RwdTransProb.from_grid_world(grid_world)
    state_trans_prob = StateTransProb.from_grid(N, M, num_actions, delta_acts)
    solver = BOESolver(gamma, state_trans_prob, rwd_trans_prob)
    v_star, pi_star = solver.solve()
    print(v_star.reshape(N,M))
    grid_world.draw_policy(pi_star, style='text', out_file=Path(f'./test_BOESolver_{inspect.currentframe().f_code.co_name}_policy.png'))
    grid_world.draw_v_pi(v_star, out_file=Path(f'./test_BOESolver_{inspect.currentframe().f_code.co_name}_v_pi.png'))
    assert np.allclose(v_star.reshape(N,M), np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
    ]), atol=1e-1)


def test_case_4():
    np.random.seed(0)
    N, M = 5, 5

    maps = np.array([
        [' ', ' ', ' ', ' ', ' '],
        [' ', '#', '#', ' ', ' '],
        [' ', ' ', '#', ' ', ' '],
        [' ', '#', 'O', '#', ' '],
        [' ', '#', ' ', ' ', ' '],
    ])
    grid_world = GridWorld.from_maps(maps)
    gamma = 0.9
    
    rwd_trans_prob = RwdTransProb.from_grid_world(grid_world, rewards=[-10, 0, 1])
    state_trans_prob = StateTransProb.from_grid(N, M, num_actions, delta_acts)
    solver = BOESolver(gamma, state_trans_prob, rwd_trans_prob)
    v_star, pi_star = solver.solve()
    print(v_star.reshape(N,M))
    grid_world.draw_policy(pi_star, style='arrows', out_file=Path(f'./test_BOESolver_{inspect.currentframe().f_code.co_name}_policy.png'))
    grid_world.draw_v_pi(v_star, out_file=Path(f'./test_BOESolver_{inspect.currentframe().f_code.co_name}_v_pi.png'))
    assert np.allclose(v_star.reshape(N,M), np.array([
        [ 3.5, 3.9, 4.3, 4.8, 5.3],
        [ 3.1, 3.5, 4.8, 5.3, 5.9],
        [ 2.8, 2.5,10.0, 5.9, 6.6],
        [ 2.5,10.0,10.0,10.0, 7.3],
        [ 2.3, 9.0,10.0, 9.0, 8.1],
    ]), atol=1e-1)


def test_case_5():
    np.random.seed(0)
    N, M = 2, 2

    maps = np.array([
        [' ', ' '],
        [' ', 'O'],
    ])
    grid_world = GridWorld.from_maps(maps)
    gamma = 0.9
    
    rwd_trans_prob = RwdTransProb.from_grid_world(grid_world, rewards=[-10, 0, 1])
    state_trans_prob = StateTransProb.from_grid(N, M, num_actions, delta_acts)
    solver = BOESolver(gamma, state_trans_prob, rwd_trans_prob)
    v_star, pi_star = solver.solve()
    print(v_star.reshape(N,M))
    grid_world.draw_policy(pi_star, style='arrows', out_file=Path(f'./test_BOESolver_{inspect.currentframe().f_code.co_name}_policy.png'))
    grid_world.draw_v_pi(v_star, out_file=Path(f'./test_BOESolver_{inspect.currentframe().f_code.co_name}_v_pi.png'))
    assert np.allclose(v_star.reshape(N,M), np.array([
        [ 9.0, 10.0],
        [10.0, 10.0],
    ]), atol=1e-1)


