from typing import List, Optional
from Action import Action


class State:
    def __init__(self, id: int, actions: Optional[List[Action]] = None):
        self.id = id
        self.actions = actions
    

    def __str__(self):
        return f"State {self.id}"
    

    def __repr__(self):
        return f"State({self.id})"
    

    def __eq__(self, other):
        return isinstance(other, State) and self.id == other.id
    

    def __index__(self) -> int:
        return self.id
    

    def __hash__(self):
        return hash(self.id)













