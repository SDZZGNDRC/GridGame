import sys
import os

# 将项目文件所在的目录添加到Python的搜索路径中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import inspect
from pathlib import Path
from MCSolver import MCSolver
from MCInterface import MCInterface
from GridWorld import GridWorld
from StateTransProb import StateTransProb
from RwdTransProb import RwdTransProb
from Action import delta_acts, num_actions


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

    mci = MCInterface(
        RwdTransProb.from_grid_world(grid_world),
        StateTransProb.from_grid(N, M, num_actions, delta_acts)
    )
    solver = MCSolver(grid_world, mci)
    pi_star = solver.solve(gamma)
    grid_world.draw_policy(pi_star, out_file=Path(f'./test_MCSolver_{inspect.currentframe().f_code.co_name}_policy.png'))



if __name__ == '__main__':
    test_case_1()









