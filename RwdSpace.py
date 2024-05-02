from typing import List, Tuple, Dict
from State import State
from Action import Action

class RwdSpace:
    def __init__(self) -> None:
        self.__spaces: Dict[Tuple[State, Action], set[float]] = {}

    def __setitem__(self, key: Tuple[State, Action], value: set[float]) -> None:
        self.__spaces[key] = value

    def __getitem__(self, key: Tuple[State, Action]) -> set[float]:
        return self.__spaces[key]

    def __len__(self) -> int:
        return len(self.__spaces)

    def __iter__(self) -> iter:
        return iter(self.__spaces)


