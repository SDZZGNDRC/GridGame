from typing import Tuple, Optional, Literal, List
import numpy as np
from GridGame.Pos import Pos

class GridWorld:
    def __init__(self, n: int, m: int, obstacles: List[Pos], init_pos: Optional[Pos] = None) -> None:
        self.n = n
        self.m = m
        self.obstacles = obstacles
        if any( obstacle.x < 0 or obstacle.x >= n or 
                obstacle.y < 0 or obstacle.y >= m for obstacle in obstacles):
            raise ValueError('Obstacle position out of range')
        self.init_pos = init_pos if init_pos else Pos.origin()
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

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.n, self.m)


















