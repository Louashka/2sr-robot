import numpy as np
import math
from typing import List

def __bezierTwoPoints(t: (float, int), P1: np.ndarray, P2: np.ndarray) -> np.ndarray:
    Q1 = (1 - t) * P1 + t * P2
    return Q1

def __bezierPoints(t: (float, int), points: List[np.ndarray]) -> List[np.ndarray]:
    newpoints = []
    for i1 in range(0, len(points) - 1):
        newpoints += [__bezierTwoPoints(t, points[i1], points[i1 + 1])]
    return newpoints

def __bezierPoint(t: (float, int), points: List[np.ndarray]) -> np.ndarray:
    newpoints = points
    while len(newpoints) > 1:
        newpoints = __bezierPoints(t, newpoints)

    return newpoints[0]

def __bezierCurve(t_values: list, points: List[np.ndarray]) -> List[np.ndarray]:
    curve = np.array([[0.0] * len(points[0])])
    for t in t_values:
        curve = np.append(curve, [__bezierPoint(t, points)], axis=0)
    curve = np.delete(curve, 0, 0)
    
    return curve

def generateCurve(start_point: List[float], lim=[[-2, 2], [-2, 2]]) -> tuple[np.ndarray, np.ndarray]:
    xlim = lim[0]
    ylim = lim[1]
    cp = [start_point]
    ncp = 3 # Number of control points
    
    while len(cp) < ncp:
        x = np.random.rand() * (xlim[1] - xlim[0]) + xlim[0]
        y = np.random.rand() * (ylim[1] - ylim[0]) + ylim[0]
        cp.append([x, y])

    cp = np.array(cp)
    # cp = np.array([[0, 2], [2, 8], [6, 6], [4, 4], [2, 2], [6, 0], [8, 4], [10, 8], [8, 10], [6, 9]])
    t_points = np.arange(0, 1, 0.01)
    curve = __bezierCurve(t_points, cp)
    
    return curve

def distance(p1, p2):
    """
    Calculate distance
    :param p1: list, point1
    :param p2: list, point2
    :return: float, distance
    """
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return math.hypot(dx, dy)

class Trajectory:
    def __init__(self, start_point: List[float]=[0,0]) -> None:
        """
        Define a trajectory class
        :param traj_x: list, list of x position
        :param traj_y: list, list of y position
        """
        self.__generateTrajectory(start_point)
        self.last_idx = 0

    def __generateTrajectory(self, start_point: List[float]) -> None:
        traj = generateCurve(start_point)
        self.traj_x = traj[:, 0]
        self.traj_y = traj[:, 1]

    def point(self, idx) -> List[float]:
        return [self.traj_x[idx], self.traj_y[idx]]
    
    @property
    def goal(self) -> List[float]:
        return [self.traj_x[-1], self.traj_y[-1]]

    def targetPoint(self, pos: List[float], goal_radius) -> List[float]:
        """
        Get the next look ahead point
        :param pos: list, robot position
        :return: list, target point
        """
        target_idx = self.last_idx
        target_point = self.point(target_idx)
        curr_dist = distance(pos, target_point)

        while curr_dist < goal_radius and target_idx < len(self.traj_x) - 1:
            target_idx += 1
            target_point = self.point(target_idx)
            curr_dist = distance(pos, target_point)

        self.last_idx = target_idx

        return self.point(target_idx)

