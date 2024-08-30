import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
import tkinter.font as font
import numpy as np
from Model import global_var as gv, robot2sr, agent_old
# from Controller import task_controller
from typing import List
from scipy.interpolate import splprep, splev
from circle_fit import taubinSVD

styles = {
    'original': {'linestyle': '-', 'alpha': 1, 'linewidth': 2},
    'target': {'linestyle': 'dashed', 'alpha': 0.3, 'linewidth': 1}
}

class GUI:
    def __init__(self) -> None:

        self.__window = None
        self.__canvas = None
        self.__axs = None

        self.__createWindow()
        self.__defineTrackingArea()

    @property
    def window(self) -> object:
        return self.__window

    def __createWindow(self) -> None:
        self.__window = tk.Tk()

        self.__window.title('Robot motion')
        self.__window.config(background='#fafafa')
        # self.__window.geometry('1200x720')        

    def __defineTrackingArea(self) -> None:
        self.__masFrame = tk.Frame(self.__window)
        self.__masFrame.pack(expand=True)
        self.__fig, self.__ax = plt.subplots()
        self.__canvas = FigureCanvasTkAgg(self.__fig, master = self.__masFrame)
        self.__canvas.get_tk_widget().pack()

    def show(self) -> None:
        self.__ax.axis('equal')
        plt.tight_layout()
        plt.legend()

        self.__canvas.draw()

    def plotMarkers(self, markers: dict) -> None:
        for marker in markers.values():
            self.__ax.plot(marker['marker_x'], marker['marker_y'], 'bo', markersize=2)

        self.show()

    def plotPaths(self, paths: dict, area_lim: list, display='original') -> None:
        style = styles[display]

        for path in paths.values():
            path = np.array(path)
            path_x = path[:,0]
            path_y = path[:,1]

            self.__axs[0, 0].plot(path_x, path_y, style['line_type'], color='tab:blue', alpha=style['alpha'])
        
        self.__axs[0, 0].set_xlim(area_lim[0])
        self.__axs[0, 0].set_ylim(area_lim[1])

        self.show()    

    def plotAgent(self, agent: robot2sr.Robot, markers: dict):        
        self.__ax.clear()

        # Plot VS segments
        vss1 = self.arc(agent)
        plt.plot(agent.x + vss1[0], agent.y + vss1[1], '-b', lw='5')

        vss2 = self.arc(agent, 2)
        plt.plot(agent.x + vss2[0], agent.y + vss2[1], '-b', lw='5')

        # Plot VSS connectores
        vss1_conn_x = [agent.x + vss1[0][-1] - gv.L_CONN * np.cos(vss1[2]), agent.x + vss1[0][-1]]
        vss1_conn_y = [agent.y + vss1[1][-1] - gv.L_CONN * np.sin(vss1[2]), agent.y + vss1[1][-1]]
        plt.plot(vss1_conn_x, vss1_conn_y, '-k', lw='5')

        vss2_conn_x = [agent.x + vss2[0][-1], agent.x + vss2[0][-1] + gv.L_CONN * np.cos(vss2[2])]
        vss2_conn_y = [agent.y + vss2[1][-1], agent.y + vss2[1][-1] + gv.L_CONN * np.sin(vss2[2])]
        plt.plot(vss2_conn_x, vss2_conn_y, '-k', lw='5')

        # Plot a body frame
        plt.plot(agent.x, agent.y, '*r')
        # plt.arrow(robot.x, robot.y, 0.05 * np.cos(robot.theta), 0.05 * np.sin(robot.theta), width=0.005, color='red')
        plt.plot([agent.x, agent.x + 0.05 * np.cos(agent.theta)], 
                 [agent.y, agent.y + 0.05 * np.sin(agent.theta)], '-r', lw='2')

        # Plot locomotion units
        LU_outline = np.array(
            [
                [-gv.LU_SIDE/2, gv.LU_SIDE/2, gv.LU_SIDE/2, -gv.LU_SIDE/2, -gv.LU_SIDE/2],
                [gv.LU_SIDE, gv.LU_SIDE, 0, 0, gv.LU_SIDE],
            ]
        )

        rot1 = np.array([[np.cos(vss1[2]), np.sin(vss1[2])], [-np.sin(vss1[2]), np.cos(vss1[2])]])
        rot2 = np.array([[np.cos(vss2[2]), np.sin(vss2[2])], [-np.sin(vss2[2]), np.cos(vss2[2])]])

        LU1_outline = (LU_outline.T.dot(rot1)).T
        LU2_outline = (LU_outline.T.dot(rot2)).T

        LU1_outline[0, :] += vss1_conn_x[0] + gv.LU_SIDE * (np.sin(vss1[2]) - np.cos(vss1[2]) / 2)
        LU1_outline[1, :] += vss1_conn_y[0] - gv.LU_SIDE * (np.cos(vss1[2]) + np.sin(vss1[2]) / 2)

        LU2_outline[0, :] += vss2_conn_x[-1] + gv.LU_SIDE * (np.sin(vss2[2]) + np.cos(vss2[2]) / 2)
        LU2_outline[1, :] += vss2_conn_y[-1] - gv.LU_SIDE * (np.cos(vss2[2]) - np.sin(vss2[2]) / 2)

        plt.plot(np.array(LU1_outline[0, :]).flatten(), np.array(LU1_outline[1, :]).flatten(), '-k')
        plt.plot(np.array(LU2_outline[0, :]).flatten(), np.array(LU2_outline[1, :]).flatten(), '-k')

        plt.plot(agent.head.x, agent.head.y, '*k')
        plt.plot([agent.head.x, agent.head.x + 0.1 * np.cos(agent.head.theta)], [agent.head.y, agent.head.y + 0.1 * np.sin(agent.head.theta)], '-k')

        plt.plot(agent.tail.x, agent.tail.y, '*k')
        plt.plot([agent.tail.x, agent.tail.x + 0.1 * np.cos(agent.tail.theta)], [agent.tail.y, agent.tail.y + 0.1 * np.sin(agent.tail.theta)], '-k')

        self.plotMarkers(markers)

        self.show()

    def arc(self, config: list, seg=1) -> tuple[np.ndarray, np.ndarray, float]:
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
    
    def plot_config(self, config: list, stiffness: list, label:str, stl:str = 'original') -> None:
        # Plot a body frame
        plt.plot(config[0], config[1], '*r')

        # Define colors based on stiffness
        colors = ['blue' if s == 0 else 'red' if s == 1 else 'black' for s in stiffness]

        # Plot VS segments
        for i in range(2):  # Assuming there are always 2 segments
            vss = self.arc(config, i+1)
            
            # Create a new style dictionary that includes the color
            segment_style = styles[stl].copy()
            segment_style['color'] = colors[i]

            plt.plot(config[0] + vss[0], config[1] + vss[1], label=label, **segment_style)

    def scatter(self, points_x: list, points_y: list, label:str) -> None:
        plt.plot(points_x, points_y, '.', label=label)

    def clear(self) -> None:
        self.__ax.clear()
