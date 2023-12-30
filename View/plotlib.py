import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
import tkinter.font as font
import numpy as np
from Model import agent_old, global_var
from Controller import task_controller

styles = {'original': {'line_type': '-', 'alpha': 1}, 'target': {'line_type': '.', 'alpha': 0.3}}

class GUI:
    def __init__(self, task: task_controller.Task) -> None:
        self.__task = task

        self.__window = None
        self.__canvas = None
        self.__axs = None

        self.__createWindow()
        self.__createSideBar()
        self.__defineTrackingArea()

    @property
    def window(self) -> object:
        return self.__window

    def __createWindow(self) -> None:
        self.__window = tk.Tk()

        self.__window.title('Robot motion')
        self.__window.config(background='#fafafa')
        # self.__window.geometry('1200x720')        

    def __createSideBar(self) -> None:
        self.__side_bar_frame = tk.Frame(self.__window)
        self.__side_bar_frame.pack(fill=tk.Y, side=tk.RIGHT)

        self.__buttons_frame = tk.Frame(self.__side_bar_frame)
        self.__buttons_frame.pack(fill=tk.NONE, expand=True)

        self.__buttons = {}
        
        self.__buttons['gen_path'] = tk.Button(self.__buttons_frame, text='Generate paths', command=self.__task.generatePaths)
        self.__buttons['start'] = tk.Button(self.__buttons_frame, text='Start', command=self.__task.start)
        self.__buttons['stop'] = tk.Button(self.__buttons_frame, text='Stop', command=self.__task.stop)
        self.__buttons['exit'] = tk.Button(self.__buttons_frame, text='Exit', command=self.__task.quit)
        
        for button in self.__buttons.values():
            button['font'] = font.Font(size=26)
            button.pack(fill=tk.X, padx=20, pady=8)

    def __defineTrackingArea(self) -> None:
        self.__masFrame = tk.Frame(self.__window)
        self.__masFrame.pack(expand=True)
        self.__fig, self.__axs = plt.subplots(nrows=2, ncols=2)
        self.__canvas = FigureCanvasTkAgg(self.__fig, master = self.__masFrame)
        self.__canvas.get_tk_widget().pack()

    def __show(self) -> None:
        for row in self.__axs:
            for val in row:
                val.axis('equal')
        plt.tight_layout()

        self.__canvas.draw()

    def plotMarkers(self, markers: dict) -> None:
        self.__axs[0, 1].clear()
        for marker in markers.values():
            self.__axs[0, 1].plot(marker['marker_x'], marker['marker_y'], 'bo', markersize=2)

        self.__show()

    def plotPaths(self, paths: dict, area_lim: list, display='original') -> None:
        style = styles[display]

        for path in paths.values():
            path = np.array(path)
            path_x = path[:,0]
            path_y = path[:,1]

            self.__axs[0, 0].plot(path_x, path_y, style['line_type'], color='tab:blue', alpha=style['alpha'])
        
        self.__axs[0, 0].set_xlim(area_lim[0])
        self.__axs[0, 0].set_ylim(area_lim[1])

        self.__show()    

    def plotAgent(self, robot: agent_old.Robot, display='original'):        

        self.__plotLU(robot.pose, robot.head)
        self.__plotLU(robot.pose, robot.tail)
        self.__plotVSF(robot.vsf)
        self.__plotFrame(robot.pose)

        self.__show()

    def __plotLU(self, robot_pose: list, lu: agent_old.LU, display='original'):
        style = styles[display]

        lu_block_rect = (lu.x + global_var.LU_R * np.cos(lu.theta + global_var.LU_ALPHA), lu.y + global_var.LU_R * np.sin(lu.theta + global_var.LU_ALPHA))
        lu_block = Rectangle(lu_block_rect, global_var.LU_SIDE, global_var.LU_SIDE, angle=np.degrees(lu.theta), edgecolor='black', facecolor='none')
        self.__axs[0, 1].add_patch(lu_block)
        # self.__ax.plot([lu.x, lu.x + 0.03 * np.cos(lu.theta)], [lu.y, lu.y + 0.03 * np.sin(lu.theta)], 'r')

        for wheel in lu.wheels:
            w_x, w_y = self.__wheelsToGlobal(robot_pose, wheel.position)
            self.__axs[0, 1].plot(w_x, w_y, 'o', color='orange', markersize=7)
            self.__axs[0, 1].plot([w_x, w_x + 0.015 * np.cos(wheel.theta+robot_pose[2])], [w_y, w_y + 0.015 * np.sin(wheel.theta+robot_pose[2])], 'r')

    def __plotVSF(self, vsf: agent_old.VSF, display='original'):
        style = styles[display]

        # self.__axs[0, 1].plot(vsf.markers_line[0], vsf.markers_line[1])
        pass

    def __plotFrame(self, frame: list, display='original'):
        style = styles[display]

        self.__axs[0, 1].plot(frame[0], frame[1], 'r*', alpha=style['alpha'])
        self.__axs[0, 1].plot([frame[0], frame[0] + 0.03 * np.cos(frame[2])], [frame[1], frame[1] + 0.03 * np.sin(frame[2])], 'r', alpha=style['alpha'])

    def __wheelsToGlobal(self, robot_pose: list, wheel: list):
        R_ob = np.array([[np.cos(robot_pose[2]), -np.sin(robot_pose[2])],
                        [np.sin(robot_pose[2]), np.cos(robot_pose[2])]])
        
        T_ob = np.block([[R_ob, np.array([robot_pose[:2]]).T], [np.zeros((1,2)), 1]])
        wheel_b = wheel + [1]

        wheel_global = np.matmul(T_ob, wheel_b).T[:-1]

        return wheel_global
    
    def __generateArc(self, config, seg: int):
        s = np.linspace(0, global_var.L_VSS, 50)

        flag = -1 if seg == 1 else 1

        gamma_array = config[2] + flag * config[2 + seg] * s

        x_0 = config[0] + flag * np.cos(config[2]) * global_var.L_CONN / 2
        y_0 = config[1] + flag * np.sin(config[2]) * global_var.L_CONN / 2

        if config[2 + seg] == 0:
            x = x_0 + [0, flag * global_var.L_VSS * np.cos(q[2])]
            y = y_0 + [0, flag * global_var.L_VSS * np.sin(q[2])]
        else:
            x = x_0 + np.sin(gamma_array) / \
                config[2 + seg] - np.sin(config[2]) / config[2 + seg]
            y = y_0 - np.cos(gamma_array) / \
                config[2 + seg] + np.cos(q[2]) / config[2 + seg]

        return [x, y]
    
    def clear(self) -> None:
        for row in self.__axs:
            for val in row:
                val.clear()

    def plotMAS(self) -> None:
        pass