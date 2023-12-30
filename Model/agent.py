from Model.frame import Frame
from Model import global_var as gv
from typing import List
import numpy as np

class Robot(Frame):
    def __init__(self, id: int, x: float, y: float, theta: float, k: List[float]=[0.0, 0.0], stiff: List[int]=[0, 0]):
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
        self.k = k
        self.stiff = stiff

    @property
    def id(self) -> int:
        return self.__id
    
    @property
    def config(self) -> np.ndarray:
        return np.array(self.pose + self.k)
    
    @config.setter
    def config(self, value) -> None:
        self.x = value[0]
        self.y = value[1]
        self.theta = value[2]
        self.k[0] = value[3]
        self.k[1] = value[4]
    
    def update(self, vel: np.ndarray, time_step: float=0.1) -> None:
        q_dot = np.dot(self.jacobain, vel)
        self.config += q_dot * time_step

    @property
    def jacobain(self) -> np.ndarray:
        # RIGID STATE

        # Checks if both segments are rigid
        flag_rigid = int(not (self.stiff[0] or self.stiff[1])) 
        J_rigid = flag_rigid * np.array([[np.cos(self.theta), -np.sin(self.theta), 0],
                                        [np.sin(self.theta), np.cos(self.theta), 0],
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