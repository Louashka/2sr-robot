from typing import List

class Frame:
    def __init__(self, pose: List[float]) -> None:
        self.x = pose[0]
        self.y = pose[1]
        self.theta = pose[2]
    
    @property
    def x(self) -> float:
        return self.__x
    
    @x.setter
    def x(self, value: float) -> None:
        self.__x = value

    @property
    def y(self) -> float:
        return self.__y
    
    @y.setter
    def y(self, value: float) -> None:
        self.__y = value

    @property
    def theta(self) -> float:
        return self.__theta
    
    @theta.setter
    def theta(self, value: float) -> None:
        self.__theta = value

    @property
    def position(self) -> List[float]:
        return [self.x, self.y]
    
    @position.setter
    def position(self, value: List[float]) -> None:
        if len(value) != 2:
            raise ValueError("Wrong number of position coordinates!")
        self.__x, self.__y = value

    @property
    def pose(self) -> List[float]:
        return [self.x, self.y, self.theta]
    
    @pose.setter
    def pose(self, value: List[float]) -> None:
        if len(value) != 3:
            raise ValueError("Wrong number of pose coordinates!")
        self.__x, self.__y, self.__theta = value