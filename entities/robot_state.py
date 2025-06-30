from entities.coordinate_frame import Frame
from typing import List
import numpy as np

class Model(Frame):
    def __init__(self, id: int, x, y, theta, k1 = 0, k2 = 0, stiffness: List[int]=[0, 0]):
        """
        Initializes the 2SR robot

        :param id: A unique identifier for the robot
        :param x: float, x position of the body frame
        :param y: float, y position of the body frame
        :param theta: float, orientation of the body frame
        :param k1: float, curvature of the first segment
        :param k2: float, curvature of the second segment
        :param stiffness: list[float], stiffness values for the segments
        """
        self.__id = id
        super().__init__(x, y, theta)

        self.k1 = k1
        self.k2 = k2

        self.t1 = None
        self.t2 = None

        self.stiff1 = stiffness[0]
        self.stiff2 = stiffness[1]

        self.head = Frame(0, 0, 0)
        self.tail = Frame(0, 0, 0)

    @property
    def id(self) -> int:
        return self.__id
    
    @property
    def k1(self) -> float:
        return self.__k1
    
    @k1.setter
    def k1(self, value) -> None:
        self.__k1 = value
        
    @property
    def k2(self) -> float:
        return self.__k2
    
    @k2.setter
    def k2(self, value) -> None:
        self.__k2 = value

    @property
    def t1(self) -> float:
        return self.__t1
    
    @t1.setter
    def t1(self, value) -> None:
        self.__t1 = value

    @property
    def t2(self) -> float:
        return self.__t2
    
    @t2.setter
    def t2(self, value) -> None:
        self.__t2 = value

    @property
    def temp(self) -> List[float]:
        return [self.t1, self.t2]
    
    @temp.setter
    def temp(self, value) -> None:
        if len(value) != 2:
            raise ValueError("Wrong number of temperature values!")
        self.__t1, self.__t2 = value

    @property
    def stiff1(self) -> float:
        return self.__stiff1
    
    @stiff1.setter
    def stiff1(self, value) -> None:
        self.__stiff1 = value

    @property
    def stiff2(self) -> float:
        return self.__stiff2
    
    @stiff2.setter
    def stiff2(self, value) -> None:
        self.__stiff2 = value

    @property
    def stiffness(self) -> List[int]:
        return [self.stiff1, self.stiff2]
    
    @stiffness.setter
    def stiffness(self, value) -> None:
        if len(value) != 2:
            raise ValueError("Wrong number of stiffness values!")
        self.__stiff1, self.__stiff2 = value

    @property
    def head(self) -> Frame:
        return self.__head
    
    @head.setter
    def head(self, value) -> None:
        self.__head = value

    @property
    def tail(self) -> Frame:
        return self.__tail
    
    @tail.setter
    def tail(self, value) -> None:
        self.__tail = value

    @property
    def curvature(self) -> List[float]:
        return [self.k1, self.k2]
    
    @property
    def config(self) -> np.ndarray:
        return np.array(self.pose + self.curvature)
    
    @config.setter
    def config(self, value) -> None:
        self.x = value[0]
        self.y = value[1]
        self.theta = value[2]
        self.k1 = value[3]
        self.k2 = value[4]
