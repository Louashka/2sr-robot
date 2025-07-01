import matplotlib.axes
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from entities import robot_state, global_var as gv

class RobotPlot:
    def __init__(self) -> None:
        self.vss_lw = 5

    def __arc(self, config: list, seg=1) -> tuple[np.ndarray, np.ndarray, float]:
        k = config[2+seg]
        l = np.linspace(0, gv.L_VSS, 50)
        flag = -1 if seg == 1 else 1
        theta_array = config[2] + flag * k * l

        if k == 0:
            x = np.array([0, flag * gv.L_VSS * np.cos(config[2])])
            y = np.array([0, flag * gv.L_VSS * np.sin(config[2])])
        else:
            x = np.sin(theta_array) / k - np.sin(config[2]) / k
            y = -np.cos(theta_array) / k + np.cos(config[2]) / k

        x += config[0]
        y += config[1]
        theta_end = theta_array[-1]
            
        return x, y, theta_end % (2 * np.pi)

    def plot_robot(self, ax: matplotlib.axes.Axes, robot: robot_state.Model):
        # Plot VS segments
        vss1 = self.__arc(robot)
        vss2 = self.__arc(robot, 2)

        ax.plot(vss1[0], vss1[1], '-b', lw=self.vss_lw)
        ax.plot(vss2[0], robot.y + vss2[1], '-b', lw=self.vss_lw)

    def plot_body_frame():
        pass