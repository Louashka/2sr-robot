import numpy as np
from Model import global_var as gv

def getDistance(p1, p2):
    """
    Calculate distance
    :param p1: list, point1
    :param p2: list, point2
    :return: float, distance
    """
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return np.hypot(dx, dy)

class LogSpiral:
    def __init__(self, n: int):
        self.m = gv.M[n-1]
        self.a = gv.SPIRAL_COEF[0][n-1]
        self.b = gv.SPIRAL_COEF[1][n-1]
        self.theta_min = gv.SPIRAL_TH_MIN[n-1]

    def get_theta(self, k: float) -> float:
        return k * gv.L_VSS / self.m

    def get_rho(self, k: float) -> float:
        b = -self.b if k > 0 else self.b
        theta = self.get_theta(k)
        theta = theta - self.theta_min if k > 0 else theta + self.theta_min
        return self.a * np.exp(b * theta)
    
    def get_pos_dot(self, theta_0: float, k: float, seg: int = 1, lu: int = 1) -> list:
        seg_flag = -1 if seg == 1 else 1
        lu_flag = -1 if lu == 1 else 1
        b = -self.b if k > 0 else self.b

        theta = self.get_theta(k)
        phi = theta_0 + seg_flag * k * gv.L_VSS
  
        pos_dot_local = np.array([
            [b * np.cos(theta) - np.sin(theta)],
            [lu_flag * (b * np.sin(theta) + np.cos(theta))]
        ])

        rho = self.get_rho(k)
        pos_dot_local *= (rho - gv.L_VSS) / rho
    
        rot_spiral_to_global = np.array([
            [np.cos(phi), -np.sin(phi)],
            [np.sin(phi), np.cos(phi)]
        ])
    
        return (rot_spiral_to_global @ pos_dot_local).flatten().tolist()
    
    def get_th_dot(self, k:float):
        return self.m / self.get_rho(k)
    
    def get_k_dot(self, k: float):
        return self.get_th_dot(k) / gv.L_VSS

class Trajectory:
    def __init__(self, traj_x, traj_y):
        """
        Define a trajectory class
        :param traj_x: list, list of x position
        :param traj_y: list, list of y position
        """
        self.x = traj_x
        self.y = traj_y
        self.yaw = []

        for i in range(len(traj_x)):
            self.yaw.append(self.getSlopeAngle(i))

        self.__calculate_cumulative_length()
        self.s = list(np.linspace(0, self.length[-1], len(traj_x)))

        self.last_idx = 0

    @property
    def params(self) -> tuple[list, list, list, list]:
        return self.x, self.y, self.yaw, self.s

    def __calculate_cumulative_length(self) -> None:
        dx = np.diff(self.x)
        dy = np.diff(self.y)
        segment_lengths = np.hypot(dx, dy)
        self.length = np.concatenate(([0], np.cumsum(segment_lengths)))

    def getPoint(self, idx) -> list:
        return [self.x[idx], self.y[idx]]

    def getTargetPoint(self, pos, la_dist) -> list:
        """
        Get the next look ahead point
        :param pos: list, vehicle position
        :return: list, target point
        """
        target_idx = self.last_idx
        target_point = self.getPoint(target_idx)
        current_dist = getDistance(pos, target_point)

        while current_dist < la_dist and target_idx < len(self.x) - 1:
            target_idx += 1
            target_point = self.getPoint(target_idx)
            current_dist = getDistance(pos, target_point)

        self.last_idx = target_idx
        return self.getPoint(target_idx)
    
    def divideIntoThirds(self) -> tuple:
        """
        Divide the path into three equal pieces and return the indices of the dividing points
        :return: tuple, (index1, index2)
        """
        total_length = self.length[-1]
        third_length = total_length / 3

        index1 = np.searchsorted(self.length, third_length)
        index2 = np.searchsorted(self.length, 2 * third_length)

        return index1, index2
    
    def getSlopeAngle(self, idx) -> float:
        """
        Calculate the angle of the slope at a given index
        :param idx: int, index of the point
        :return: float, angle in radians from -pi to pi
        """
        if idx == 0:
            dx = self.x[1] - self.x[0]
            dy = self.y[1] - self.y[0]
        elif idx == len(self.x) - 1:
            dx = self.x[-1] - self.x[-2]
            dy = self.y[-1] - self.y[-2]
        else:
            dx = self.x[idx+1] - self.x[idx-1]
            dy = self.y[idx+1] - self.y[idx-1]

        return np.arctan2(dy, dx)