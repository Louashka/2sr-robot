import numpy as np

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

class Trajectory:
    def __init__(self, traj_x, traj_y):
        """
        Define a trajectory class
        :param traj_x: list, list of x position
        :param traj_y: list, list of y position
        """
        self.traj_x = traj_x
        self.traj_y = traj_y
        self.last_idx = 0
        self.__calculate_cumulative_length()

    def __calculate_cumulative_length(self) -> None:
        dx = np.diff(self.traj_x)
        dy = np.diff(self.traj_y)
        segment_lengths = np.sqrt(dx**2 + dy**2)
        self.length = np.concatenate(([0], np.cumsum(segment_lengths)))

    def getPoint(self, idx) -> list:
        return [self.traj_x[idx], self.traj_y[idx]]

    def getTargetPoint(self, pos, la_dist) -> list:
        """
        Get the next look ahead point
        :param pos: list, vehicle position
        :return: list, target point
        """
        target_idx = self.last_idx
        target_point = self.getPoint(target_idx)
        current_dist = getDistance(pos, target_point)

        while current_dist < la_dist and target_idx < len(self.traj_x) - 1:
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
            dx = self.traj_x[1] - self.traj_x[0]
            dy = self.traj_y[1] - self.traj_y[0]
        elif idx == len(self.traj_x) - 1:
            dx = self.traj_x[-1] - self.traj_x[-2]
            dy = self.traj_y[-1] - self.traj_y[-2]
        else:
            dx = self.traj_x[idx+1] - self.traj_x[idx-1]
            dy = self.traj_y[idx+1] - self.traj_y[idx-1]

        return np.arctan2(dy, dx)