from typing import List, Dict, Tuple
from State import State
from Action import Action

class ActionSpace:
    def __init__(self):
        self.__spaces: Dict[State, set[Action]] = {}

    def __setitem__(self, state: State, actions: set[Action]) -> None:
        self.__spaces[state] = actions

    def __getitem__(self, state: State) -> set[Action]:
        return self.__spaces[state]

    def __contains__(self, state: State) -> bool:
        return state in self.__spaces











