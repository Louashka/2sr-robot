import numpy as np
import matplotlib.pyplot as plt
from entities import global_var as gv


def arc(config: list, seg=1) -> tuple[np.ndarray, np.ndarray, float]:
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

        theta_end = theta_array[-1]
            
        return x, y, theta_end % (2 * np.pi)


