from typing import List, Optional

class Pos:
    '''
    A class to represent a position on the grid.
    '''
    @classmethod
    def origin(cls) -> "Pos":
        return cls(0, 0)
    

    @classmethod
    def from_list(cls, lst: list) -> List["Pos"]:
        return [cls(x, y) for x, y in lst]

    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y

    def __list__(self) -> list:
        return [self.x, self.y]

    def __tuple__(self) -> tuple:
        return (self.x, self.y)

    def __str__(self) -> str:
        return f"({self.x}, {self.y})"
    

    def __repr__(self) -> str:
        return f"Pos({self.x}, {self.y})"
    

    def __eq__(self, other) -> bool:
        if isinstance(other, Pos):
            return self.x == other.x and self.y == other.y
        if isinstance(other, tuple) and len(other) == 2:
            return self.x == other[0] and self.y == other[1]
        else:
            raise TypeError("unsupported operand type(s) for ==: 'Pos' and '{}'".format(type(other)))
    

    def __hash__(self) -> int:
        return hash((self.x, self.y))
    

    def __add__(self, other) -> "Pos":
        return Pos(self.x + other.x, self.y + other.y)
    

    def __sub__(self, other) -> "Pos":
        return Pos(self.x - other.x, self.y - other.y)
    

    def __mul__(self, other) -> "Pos":
        return Pos(self.x * other, self.y * other)
    

    def __iter__(self):
        return iter((self.x, self.y))
    







