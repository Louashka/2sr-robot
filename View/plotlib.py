import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
import tkinter.font as font
import numpy as np
from Model import global_var, agent
from Controller import task_controller

styles = {'original': {'line_type': '-', 'alpha': 1}, 'target': {'line_type': '.', 'alpha': 0.3}}

class GUI:
    def __init__(self, task: task_controller.Task) -> None:
        self.__task = task

        self.__window = None
        self.__canvas = None
        self.__ax = None

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
        self.__window.geometry('1200x720')        

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
        self.__fig, self.__ax = plt.subplots()
        self.__canvas = FigureCanvasTkAgg(self.__fig, master = self.__masFrame)
        self.__canvas.get_tk_widget().pack()

    def __show(self) -> None:
        self.__ax.axis('equal')
        plt.tight_layout()

        self.__canvas.draw()

    def plotMarkers(self, markers: dict) -> None:
        for marker in markers.values():
            self.__ax.plot(marker['marker_x'], marker['marker_y'], 'bo', markersize=2)

        self.__show()

    def plotPath(self, path: list, area_lim: list, display='original') -> None:
        style = styles[display]
        self.__ax.clear()

        path = np.array(path)
        path_x = path[:,0]
        path_y = path[:,1]

        self.__ax.plot(path_x, path_y, style['line_type'], color='tab:blue', alpha=style['alpha'])
        self.__ax.set_xlim(area_lim[0])
        self.__ax.set_ylim(area_lim[1])

        self.__canvas.draw()        

    def plotAgent(self, robot: agent.Robot, display='original'):        

        self.__plotLU(robot.head)
        self.__plotLU(robot.tail)
        self.__plotVSF(robot.vsf)
        self.__plotFrame(robot.pose)

        self.__canvas.draw()

    def __plotLU(self, lu: agent.LU, display='original'):
        style = styles[display]

        lu_block_rect = (lu.x + global_var.LU_R * np.cos(lu.theta + global_var.LU_ALPHA), lu.y + global_var.LU_R * np.sin(lu.theta + global_var.LU_ALPHA))
        lu_block = Rectangle(lu_block_rect, global_var.LU_SIDE, global_var.LU_SIDE, angle=np.degrees(lu.theta), edgecolor='black', facecolor='none')
        self.__ax.add_patch(lu_block)
        # self.__ax.plot([lu.x, lu.x + 0.03 * np.cos(lu.theta)], [lu.y, lu.y + 0.03 * np.sin(lu.theta)], 'r')

        for wheel in lu.wheels:
            # wheel_global = self.__wheelsToGlobal(lu.pose, wheel.position)
            self.__ax.plot(wheel.x, wheel.y, 'o', color='orange', markersize=7)

    def __plotVSF(self, vsf: agent.VSF, display='original'):
        style = styles[display]

        # self.__ax.plot(vsf.markers_line[0], vsf.markers_line[1])
        pass

    def __plotFrame(self, frame: list, display='original'):
        style = styles[display]

        self.__ax.plot(frame[0], frame[1], 'r*', alpha=style['alpha'])
        self.__ax.plot([frame[0], frame[0] + 0.03 * np.cos(frame[2])], [frame[1], frame[1] + 0.03 * np.sin(frame[2])], 'r', alpha=style['alpha'])

    def __wheelsToGlobal(self, robot_pose, wheel):
        R_ob = np.array([[np.cos(robot_pose[2]), -np.sin(robot_pose[2])],
                        [np.sin(robot_pose[2]), np.cos(robot_pose[2])]])
        
        T_ob = np.block([[R_ob, np.array([robot_pose[:2]]).T], [np.zeros((1,2)), 1]])
        wheel_b = wheel + [1]

        wheel_global = np.matmul(T_ob, wheel_b).T

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
    
    def __plotMarkers(self, markers):
        pass