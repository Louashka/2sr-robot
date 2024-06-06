from Model.frame import Frame
from Model import global_var as gv
from typing import List
import numpy as np

class Robot(Frame):
    def __init__(self, id: int, x, y, theta, k1, k2, stiffness: List[int]=[0, 0]):
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

    @property 
    def body_frame(self) -> np.ndarray:
        # lu_head = self.__lu_position(1)
        # lu_tail = self.__lu_position(2)

        # robot_x = (lu_head[0] + lu_tail[0]) / 2
        # robot_y = (lu_head[1] + lu_tail[1]) / 2

        # dy = lu_tail[1] - lu_head[1]
        # dx = lu_tail[0] - lu_head[0]

        robot_x = (self.head.x + self.tail.x) / 2
        robot_y = (self.head.y + self.tail.y) / 2

        dy = self.tail.y - self.head.y
        dx = self.tail.x - self.head.x

        robot_theta = np.arctan(dy/dx)
        if dx < 0:
            robot_theta -= np.pi
        
        return np.array([robot_x, robot_y, robot_theta])
    
    def __lu_position(self, id=1) -> List[float]:
        vss = self.arc(id)
        vss_end = [self.x + vss[0][-1], self.y + vss[1][-1]]

        flag = -1 if id == 1 else 1

        vss_conn_x = vss_end[0] + flag * gv.L_CONN * np.cos(vss[2])
        vss_conn_y = vss_end[1] + flag * gv.L_CONN * np.sin(vss[2])
        
        lu_x = vss_conn_x + np.sqrt(2) / 2 * gv.LU_SIDE * np.cos(vss[2] + np.pi / 2 - flag * 3 * np.pi / 4)
        lu_y = vss_conn_y + np.sqrt(2) / 2 * gv.LU_SIDE * np.sin(vss[2] + np.pi / 2 - flag * 3 * np.pi / 4)

        return [lu_x, lu_y]

    def arc(self, seg=1) -> tuple[np.ndarray, np.ndarray, float]:
        k = self.curvature[seg-1]
        l = np.linspace(0, gv.L_VSS, 50)
        flag = -1 if seg == 1 else 1
        theta_array = self.theta + flag * k * l

        if k == 0:
            x = np.array([0, flag * gv.L_VSS * np.cos(self.theta)])
            y = np.array([0, flag * gv.L_VSS * np.sin(self.theta)])
        else:
            x = np.sin(theta_array) / k - np.sin(self.theta) / k
            y = -np.cos(theta_array) / k + np.cos(self.theta) / k

        theta_end = theta_array[-1]
            
        return x, y, theta_end % (2 * np.pi)
    
    @property
    def jacobian_rigid(self) -> np.ndarray:
        body_frame = self.body_frame

        dy = self.y - body_frame[1]
        dx = self.x - body_frame[0]

        rho_ab = np.hypot(dx, dy)
        theta_ab = np.arctan(dy/dx)

        if dx < 0:
            theta_ab -= np.pi

        J_rigid = np.array([[np.cos(body_frame[2]), -np.sin(body_frame[2]), -rho_ab * np.sin(theta_ab)],
                            [np.sin(body_frame[2]), np.cos(body_frame[2]), rho_ab * np.cos(theta_ab)],
                            [0, 0, 1],
                            [0, 0, 0],
                            [0, 0, 0]])
        
        return J_rigid
    
    def update(self, v: np.ndarray, time_step: float=0.1) -> None:
        q_dot = self.jacobian_rigid.dot(v)
        self.config += q_dot * time_step
