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

    @property
    def id(self) -> int:
        return self.__id
    
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
        lu_head = self.lu_position(1)
        lu_tail = self.lu_position(2)

        robot_x = (lu_head[0] + lu_tail[0]) / 2
        robot_y = (lu_head[1] + lu_tail[1]) / 2

        dy = lu_tail[1] - lu_head[1]
        dx = lu_tail[0] - lu_head[0]

        robot_theta = np.arctan(dy/dx)
        if dx < 0:
            robot_theta -= np.pi
        
        return np.array([robot_x, robot_y, robot_theta])
    
    def lu_position(self, id=1) -> List[float]:
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

    def __calcRotFrame(self):
        vss1 = self.arc(self.theta, self.k[0])
        vss2 = self.arc(self.theta, self.k[1], 2)

        vss1_conn_x = [self.x + vss1[0][-1] - gv.L_CONN * np.cos(vss1[2]), self.x + vss1[0][-1]]
        vss1_conn_y = [self.y + vss1[1][-1] - gv.L_CONN * np.sin(vss1[2]), self.y + vss1[1][-1]]

        vss2_conn_x = [self.x + vss2[0][-1], self.x + vss2[0][-1] + gv.L_CONN * np.cos(vss2[2])]
        vss2_conn_y = [self.y + vss2[1][-1], self.y + vss2[1][-1] + gv.L_CONN * np.sin(vss2[2])]

        lu_head_x = vss1_conn_x[0] + np.sqrt(2) / 2 * gv.LU_SIDE * np.cos(vss1[2] + np.pi + np.pi / 4)
        lu_head_y = vss1_conn_y[0] + np.sqrt(2) / 2 * gv.LU_SIDE * np.sin(vss1[2] + np.pi + np.pi / 4)

        lu_tail_x = vss2_conn_x[1] + np.sqrt(2) / 2 * gv.LU_SIDE * np.cos(vss2[2] - np.pi / 4)
        lu_tail_y = vss2_conn_y[1] + np.sqrt(2) / 2 * gv.LU_SIDE * np.sin(vss2[2] - np.pi / 4)

        x = (lu_head_x + lu_tail_x) / 2
        y = (lu_head_y + lu_tail_y) / 2

        dy = lu_tail_y - lu_head_y
        dx = lu_tail_x - lu_head_x

        theta = np.arctan(dy/dx)
        if dx < 0:
            theta -= np.pi

        dy = y - self.y
        dx = x - self.x

        r = np.hypot(dx, dy)
        phi = np.arctan(dy/dx)

        if dx < 0:
            phi -= np.pi

        return r, theta, phi
    
    def update(self, v: np.ndarray, time_step: float=0.1) -> None:
        q_dot = self.jacobian_rigid.dot(v)
        self.config += q_dot * time_step

    @property
    def jacobain(self) -> np.ndarray:
        # Determine the axis of rotation
        rot_frame = self.__calcRotFrame()

        # RIGID STATE

        # Checks if both segments are rigid
        flag_rigid = int(not (self.stiff[0] or self.stiff[1])) 
        J_rigid = flag_rigid * np.array([[np.cos(rot_frame[1]), -np.sin(rot_frame[1]), rot_frame[0] * np.cos(rot_frame[2] - np.pi / 2)],
                                        [np.sin(rot_frame[1]), np.cos(rot_frame[1]), rot_frame[0] * np.sin(rot_frame[2] - np.pi / 2)],
                                        [0, 0, 1],
                                        [0, 0, 0],
                                        [0, 0, 0]])

        # SOFT STATE

        # Central angles of VSS arcs
        alpha = np.array(self.k) * gv.L_VSS
        # Angles of 3 types of logarithmic spirals
        theta = np.divide(np.tile(np.reshape(alpha, (2, 1)), 3), gv.M) - np.pi

        # Spiral constants
        a = gv.SPIRAL_COEF[0]
        b = np.tile(gv.SPIRAL_COEF[1], (2, 1))
        for i in range(2):
            if self.k[i] > 0:
                b[i] = -b[i]
                theta[i] += 2 * np.pi

        rho = a * np.exp(b * theta)  # spiral radii

        # Proportionality coefficient of the rate of VSS curvature change
        Kappa = np.divide(gv.M, gv.L_VSS * rho)
        # Proportionality coefficient of the rate of 2SRR orientation change
        Phi = np.divide(gv.M, rho)

        flag_soft1 = int(not self.stiff[1] and self.stiff[0])  # checks if VSS1 is soft
        flag_soft2 = int(not self.stiff[0] and self.stiff[1])  # checks if VSS2 is soft
        flag_full_soft = int(self.stiff[0] and self.stiff[1])  # checks if both VSSs are soft

        #  Proportionality coefficients of the rate of change of 2SRR position coordinates
        Delta = np.zeros((2, 2))
        Delta[:, 0] = (flag_soft2 * Kappa[1, 1] + flag_full_soft * Kappa[1, 2]) * \
            self.__positionDerivative(rho[1, 0], self.theta, self.k[1], self.k[1], 2).ravel()
        Delta[:, 1] = (flag_soft1 * Kappa[0, 1] + flag_full_soft * Kappa[0, 2]) * \
            self.__positionDerivative(rho[0, 0], self.theta, self.k[0], self.k[0], 1).ravel()

        J_soft = np.array([[-flag_soft2 * Phi[1, 1] - flag_full_soft * Phi[1, 2],
                            flag_soft1 * Phi[0, 1] + flag_full_soft * Phi[0, 2]],
                        [flag_soft1 * Kappa[0, 0] + flag_full_soft * Kappa[0, 2],
                            flag_soft1 * Kappa[0, 1] + flag_full_soft * Kappa[0, 2]],
                        [flag_soft2 * Kappa[1, 1] + flag_full_soft * Kappa[1, 2],
                            flag_soft2 * Kappa[1, 0] + flag_full_soft * Kappa[1, 2]]])

        J_soft = np.concatenate((Delta, J_soft), axis=0)

        # complete hybrid Jacobian matrix
        J = np.concatenate((J_soft, J_rigid), axis=1)

        return J
    
    def __positionDerivative(self, r: float, phi: float, k: float, k0: float, seg: int) -> np.ndarray:
        if seg == 1:
            pos = np.array([[-0.0266 * r * np.sin(phi + 0.0266 * k - 0.04 * k0) - 0.0006 * np.sin(phi + 0.04 * k - 0.04 * k0)],
                            [0.0266 * r * np.cos(phi + 0.0266 * k - 0.04 * k0) + 0.0006 * np.cos(phi + 0.04 * k - 0.04 * k0)]])
            # pos = bodyFramePosition(-1)
        elif seg == 2:
            pos = np.array([[-0.0266 * r * np.sin(phi - 0.0266 * k + 0.04 * k0) - 0.0006 * np.sin(phi - 0.04 * k + 0.04 * k0)],
                            [0.0266 * r * np.cos(phi - 0.0266 * k + 0.04 * k0) + 0.0006 * np.cos(phi - 0.04 * k + 0.04 * k0)]])
            
        return pos