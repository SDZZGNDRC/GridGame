from typing import Callable
from GridWorld import GridWorld


class Action:
    def __init__(self, id: str, grid_world: GridWorld, fn: Callable[[GridWorld], bool]):
        self.id = id
        self.grid_world = grid_world
        self.fn = fn

    def __call__(self) -> bool:
        return self.fn(self.grid_world)
    

    def __eq__(self, value: object) -> bool:
        return isinstance(value, Action) and self.id == value.id
    

    def __str__(self) -> str:
        return f"Action {self.id}"
    

    def __repr__(self) -> str:
        return f"Action({self.id})"
    

    def __index__(self) -> int:
        return self.id
    

    def __hash__(self) -> int:
        return hash(self.id)
    
