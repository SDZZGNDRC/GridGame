from typing import Callable, Optional
from GridWorld import GridWorld

class Action:
    def __init__(self, id: str, grid_world: Optional[GridWorld] = None, fn: Optional[Callable[[GridWorld], bool]] = None):
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


