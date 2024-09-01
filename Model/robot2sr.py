from Model.frame import Frame
from Model import splines
from typing import List
import numpy as np

class Robot(Frame):
    def __init__(self, id: int, x, y, theta, k1 = 0, k2 = 0, stiffness: List[int]=[0, 0]):
        """
        Define a robot class
        :param x: float, x position
        :param y: float, y position
        :param theta: float, robot orientation
        :param k: list, list of VSF curvature values 
        :param s: list, list of VSF stiffness values
        """
        self.__id = id
        super().__init__(x, y, theta)

        self.k1 = k1
        self.k2 = k2
        self.stiffness = stiffness

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
    def stiffness(self) -> List[int]:
        return self.__stiffness
    
    @stiffness.setter
    def stiffness(self, value) -> None:
        self.__stiffness = value

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
    
    
    def jacobian_soft(self, varsigma) -> np.ndarray:
        cardioid1 = splines.Cardioid(1)
        cardioid2 = splines.Cardioid(2)
        cardioid3 = splines.Cardioid(3)

        if all(s):
            spiral1 = spiral2 = cardioid3
        else:
            spiral1 = cardioid1
            spiral2 = cardioid2

        k1_ratio = spiral2.k_dot(q[4]) / cardioid1.k_dot(q[4])
        k2_ratio = spiral2.k_dot(q[3]) / cardioid1.k_dot(q[3])

        pos_lu1 = cardioid1.pos_dot(q[2], q[4], 2, 1)
        pos_lu2 = cardioid1.pos_dot(q[2], q[3], 1, 2)

        J = np.array([
            [k1_ratio * pos_lu1[0], k2_ratio * pos_lu2[0]],
            [k1_ratio * pos_lu1[1], k2_ratio * pos_lu2[1]], 
            [spiral2.th_dot(q[4]), spiral2.th_dot(q[3])], 
            [-spiral1.k_dot(q[3]), spiral2.k_dot(q[3])], 
            [-spiral2.k_dot(q[4]), spiral1.k_dot(q[4])]
        ])
        
        stiffness_array = np.array([[s[1], s[0]],
                                    [s[1], s[0]],
                                    [s[1], s[0]],
                                    [s[0], s[0]],
                                    [s[1], s[1]]])
        
        J_soft = np.multiply(stiffness_array, J)
        
        return J_soft
    
    def jacobian(self, varsigma) -> np.ndarray:
        return np.hstack((int(not(any(varsigma))) * self.jacobian_rigid(), self.jacobian_soft(varsigma)))
    
    def update(self, v: np.ndarray, time_step: float=0.1) -> None:
        q_dot = self.jacobian_rigid().dot(v)
        self.config += q_dot * time_step
