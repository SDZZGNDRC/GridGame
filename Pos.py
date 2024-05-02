

class Pos:
    '''
    A class to represent a position on the grid.
    '''
    @classmethod
    def origin(cls) -> "Pos":
        return cls(0, 0)
    

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
        return self.x == other.x and self.y == other.y
    

    def __hash__(self) -> int:
        return hash((self.x, self.y))
    

    def __add__(self, other) -> "Pos":
        return Pos(self.x + other.x, self.y + other.y)
    

    def __sub__(self, other) -> "Pos":
        return Pos(self.x - other.x, self.y - other.y)
    

    def __mul__(self, other) -> "Pos":
        return Pos(self.x * other, self.y * other)
    

    @property
    def neighbors(self) -> list:
        return [
            Pos(self.x + 1, self.y), Pos(self.x - 1, self.y), 
            Pos(self.x, self.y + 1), Pos(self.x, self.y - 1)
        ]
    

    @property
    def diagonal_neighbors(self) -> list:
        return [
            Pos(self.x + 1, self.y + 1), Pos(self.x - 1, self.y - 1), 
            Pos(self.x + 1, self.y - 1), Pos(self.x - 1, self.y + 1)
        ]
    






