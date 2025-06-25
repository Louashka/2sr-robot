from entities.coordinate_frame import Frame
from entities import splines
from typing import List
import numpy as np

class Model(Frame):
    def __init__(self, id: int, x, y, theta, k1 = 0, k2 = 0, stiffness: List[int]=[0, 0]):
        """
        Define a robot class
        :param x: float, x position
        :param y: float, y position
        :param theta: float, robot orientation
        :param k: list, list of VSB curvature values 
        :param s: list, list of VSB stiffness values
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

    def jacobian_rigid(self) -> np.ndarray:
        J_rigid = np.array([[np.cos(self.theta), -np.sin(self.theta), 0],
                            [np.sin(self.theta), np.cos(self.theta), 0],
                            [0, 0, 1],
                            [0, 0, 0],
                            [0, 0, 0]])
        
        return J_rigid
    
    
    def jacobian_flexible(self, varsigma) -> np.ndarray:
        cardioid1 = splines.Cardioid(1)
        cardioid2 = splines.Cardioid(2)
        cardioid3 = splines.Cardioid(3)

        if all(varsigma):
            spiral1 = spiral2 = cardioid3
        else:
            spiral1 = cardioid1
            spiral2 = cardioid2

        k1_ratio = spiral2.k_dot(self.k2) / cardioid1.k_dot(self.k2)
        k2_ratio = spiral2.k_dot(self.k1) / cardioid1.k_dot(self.k1)

        pos_lu1 = cardioid1.pos_dot(self.theta, self.k2, 2, 1)
        pos_lu2 = cardioid1.pos_dot(self.theta, self.k1, 1, 2)

        J = np.array([
            [k1_ratio * pos_lu1[0], k2_ratio * pos_lu2[0]],
            [k1_ratio * pos_lu1[1], k2_ratio * pos_lu2[1]], 
            [spiral2.th_dot(self.k2), spiral2.th_dot(self.k1)], 
            [-spiral1.k_dot(self.k1), spiral2.k_dot(self.k1)], 
            [-spiral2.k_dot(self.k2), spiral1.k_dot(self.k2)]
        ])
        
        stiffness_array = np.array([[varsigma[1], varsigma[0]],
                                    [varsigma[1], varsigma[0]],
                                    [varsigma[1], varsigma[0]],
                                    [varsigma[0], varsigma[0]],
                                    [varsigma[1], varsigma[1]]])
        
        J_soft = np.multiply(stiffness_array, J)
        
        return J_soft
    
    def jacobian(self, varsigma) -> np.ndarray:
        return np.hstack((int(not(any(varsigma))) * self.jacobian_rigid(), self.jacobian_flexible(varsigma)))
    
    def update(self, v: np.ndarray, time_step: float=0.1) -> None:
        q_dot = self.jacobian_rigid().dot(v)
        self.config += q_dot * time_step
