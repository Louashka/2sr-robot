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
        self.phi0 = gv.SPIRAL_PHI0[n-1]

    def get_phi(self, k: float) -> float:
        return k * gv.L_VSS / self.m

    def get_rho(self, k: float) -> float:
        b = -self.b if k > 0 else self.b
        phi = self.get_phi(k)
        phi = phi - self.phi0 if k > 0 else phi + self.phi0
        return self.a * np.exp(b * phi)
    
    def get_pos_dot(self, th0: float, k: float, seg: int = 1, lu: int = 1) -> list:
        b = -self.b if k > 0 else self.b

        phi = self.get_phi(k)
        phi = phi - self.phi0 if k > 0 else phi + self.phi0
  
        pos_dot_local = np.array([
            [b * np.cos(phi) - np.sin(phi)],
            [(-1)**lu * (b * np.sin(phi) + np.cos(phi))]
        ])

        rho = self.get_rho(k)
        pos_dot_local *= (rho - gv.L_VSS) / rho

        th = th0 + (-1)**seg * k * gv.L_VSS
    
        rot_spiral_to_global = np.array([
            [np.cos(th), -np.sin(th)],
            [np.sin(th), np.cos(th)]
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
            yaw = self.getSlopeAngle(i) - np.pi/2
            while(yaw > np.pi):
                yaw = yaw - 2.0 * np.pi
            while(yaw < -np.pi):
                yaw = yaw + 2.0 * np.pi
            self.yaw.append(yaw)

        self.__calculate_cumulative_length()
        self.s = list(np.linspace(0, self.length[-1], len(traj_x)))

        self.last_idx = 0
        self.curvature = self.calculate_curvature()

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

    def getTarget(self, pos, la_dist) -> int:
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
        return target_idx
    
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
    
    def calculate_curvature(self) -> list:
        """
        Calculate the curvature at each point of the trajectory
        :return: list of curvatures
        """
        curvature = []
        n = len(self.x)

        for i in range(n):
            if i == 0:
                dx1, dy1 = self.x[1] - self.x[0], self.y[1] - self.y[0]
                dx2, dy2 = self.x[2] - self.x[1], self.y[2] - self.y[1]
            elif i == n - 1:
                dx1, dy1 = self.x[-1] - self.x[-2], self.y[-1] - self.y[-2]
                dx2, dy2 = self.x[-1] - self.x[-2], self.y[-1] - self.y[-2]
            else:
                dx1, dy1 = self.x[i] - self.x[i-1], self.y[i] - self.y[i-1]
                dx2, dy2 = self.x[i+1] - self.x[i], self.y[i+1] - self.y[i]

            dx = (dx1 + dx2) / 2
            dy = (dy1 + dy2) / 2

            ddx = dx2 - dx1
            ddy = dy2 - dy1

            numerator = abs(dx * ddy - dy * ddx)
            denominator = (dx**2 + dy**2)**1.5

            if denominator == 0:
                curvature.append(0)  # Straight line
            else:
                curvature.append(numerator / denominator)

        return curvature