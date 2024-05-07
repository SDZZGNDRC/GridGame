import math
from typing import Tuple, Optional, Literal, List
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from GridGame.Pos import Pos

class GridWorld:
    def __init__(self, n: int, m: int, obstacles: List[Pos], init_pos: Optional[Pos] = None, target: Optional[Pos] = None) -> None:
        self.n = n
        self.m = m
        self.obstacles = obstacles
        if any( obstacle.x < 0 or obstacle.x >= n or 
                obstacle.y < 0 or obstacle.y >= m for obstacle in obstacles):
            raise ValueError('Obstacle position out of range')
        self.init_pos = init_pos if init_pos else Pos.origin()
        self.target = target if target else Pos(n - 1, m - 1)
        self.now_pos = self.init_pos

    def move(self, direction: Literal['up', 'down', 'left', 'right']) -> bool:
        '''
        Move the agent to the given direction.
        Return True if not in an obstacle, otherwise return False.
        '''
        if direction == 'up':
            self.now_pos = min(0, self.now_pos.y - 1)
        elif direction == 'down':
            self.now_pos = max(self.n - 1, self.now_pos.y + 1)
        elif direction == 'left':
            self.now_pos = min(0, self.now_pos.x - 1)
        elif direction == 'right':
            self.now_pos = max(self.m - 1, self.now_pos.x + 1)
        else:
            raise ValueError('Invalid direction')
        if self.now_pos in self.obstacles:
            return False
        return True

    def neighbors(self, pos: Pos) -> List[Pos]:
        # check if the position is within the grid
        neighbors = []
        if pos.x < 0 or pos.x >= self.m or pos.y < 0 or pos.y >= self.n:
            return neighbors
        if pos.x > 0:
            neighbors.append(Pos(pos.x - 1, pos.y))
        if pos.x < self.m - 1:
            neighbors.append(Pos(pos.x + 1, pos.y))
        if pos.y > 0:
            neighbors.append(Pos(pos.x, pos.y - 1))
        if pos.y < self.n - 1:
            neighbors.append(Pos(pos.x, pos.y + 1))
        return neighbors

    def diag_neighbors(self, pos: Pos) -> List[Pos]:
        # check if the position is within the grid
        neighbors = []
        if pos.x < 0 or pos.x >= self.m or pos.y < 0 or pos.y >= self.n:
            return neighbors
        if pos.x > 0 and pos.y > 0:
            neighbors.append(Pos(pos.x - 1, pos.y - 1))
        if pos.x < self.m - 1 and pos.y > 0:
            neighbors.append(Pos(pos.x + 1, pos.y - 1))
        if pos.x > 0 and pos.y < self.n - 1:
            neighbors.append(Pos(pos.x - 1, pos.y + 1))
        if pos.x < self.m - 1 and pos.y < self.n - 1:
            neighbors.append(Pos(pos.x + 1, pos.y + 1))
        return neighbors

    def draw_policy(self, pi: np.ndarray, style: Literal['arrows', 'text'] = 'arrows', out_file: Optional[Path] = None) -> None:
        arrow_length = 0.7
        _, axs = plt.subplots(self.n, self.m, figsize=(5, 5))
        for i in range(self.n):
            for j in range(self.m):
                if (i, j) in self.obstacles:
                    color = 'orange'
                    rect = Rectangle((-1, -1), 2, 2, color=color)
                    axs[i, j].add_patch(rect)
                elif (i, j) == self.target:
                    color = 'blue'
                    rect = Rectangle((-1, -1), 2, 2, color=color)
                    axs[i, j].add_patch(rect)
                
                s = i*self.n + j

                if style == 'arrows':
                    if np.allclose(pi[s, :-1], 0):
                        # draw a circle to indicate the agent cannot move
                        circle = Circle((0, 0), 0.2, color='green', fill=False)
                        axs[i, j].add_patch(circle)
                    else:
                        for d, (dx, dy) in enumerate([(0, 1), (0, -1), (-1, 0), (1, 0)]):
                            if math.isclose(pi[s, d], 0):
                                continue
                            axs[i, j].arrow(0, 0, dx*pi[s, d]*arrow_length, dy*pi[s, d]*arrow_length, head_width=0.1, head_length=0.1, fc='green', ec='green')
                elif style == 'text':
                    axs[i, j].text(-0.5, 0, str(pi[s, 2]), ha='center', va='center')
                    axs[i, j].text(0.5, 0, str(pi[s, 3]), ha='center', va='center')
                    axs[i, j].text(0, -0.5, str(pi[s, 1]), ha='center', va='center')
                    axs[i, j].text(0, 0.5, str(pi[s, 0]), ha='center', va='center')
                else:
                    raise ValueError('Invalid style')

                axs[i, j].set_xlim(-1, 1)
                axs[i, j].set_ylim(-1, 1)
                axs[i, j].spines['left'].set_linewidth(0.5)
                axs[i, j].spines['right'].set_linewidth(0.5)
                axs[i, j].spines['bottom'].set_linewidth(0.5)
                axs[i, j].spines['top'].set_linewidth(0.5)
                axs[i, j].set_xticks([])
                axs[i, j].set_yticks([])

        plt.subplots_adjust(wspace=0, hspace=0)
        if out_file:
            plt.savefig(out_file)
        else:
            plt.show()

    def draw_v_pi(self, v_pi: np.ndarray, decimal_places: int = 1, out_file: Optional[Path] = None) -> None:
        _, axs = plt.subplots(self.n, self.m, figsize=(5, 5))
        for i in range(self.n):
            for j in range(self.m):
                if (i, j) in self.obstacles:
                    color = 'orange'
                    rect = Rectangle((-1, -1), 2, 2, color=color)
                    axs[i, j].add_patch(rect)
                elif (i, j) == self.target:
                    color = 'blue'
                    rect = Rectangle((-1, -1), 2, 2, color=color)
                    axs[i, j].add_patch(rect)
                
                s = i*self.n + j

                v_pi_rounded = round(v_pi[s], decimal_places)
                axs[i, j].text(0, 0, str(v_pi_rounded), ha='center', va='center')

                axs[i, j].set_xlim(-1, 1)
                axs[i, j].set_ylim(-1, 1)
                axs[i, j].spines['left'].set_linewidth(0.5)
                axs[i, j].spines['right'].set_linewidth(0.5)
                axs[i, j].spines['bottom'].set_linewidth(0.5)
                axs[i, j].spines['top'].set_linewidth(0.5)
                axs[i, j].set_xticks([])
                axs[i, j].set_yticks([])

        plt.subplots_adjust(wspace=0, hspace=0)
        if out_file:
            plt.savefig(out_file)
        else:
            plt.show()

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.n, self.m)

    @classmethod
    def from_maps(cls, maps: np.ndarray) -> 'GridWorld':
        N, M = maps.shape
        obstacles = []
        target = None
        for i in range(N):
            for j in range(M):
                if maps[i][j] == '#':
                    obstacles.append(Pos(i,j))
                elif maps[i][j] == 'O':
                    target = Pos(i,j)
        return cls(N, M, obstacles, target=target)
















