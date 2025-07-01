import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
import tkinter.font as font
import numpy as np
from model import global_var as gv, robot2sr
from controller import func
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

    def plot(self, x, y) -> None:
        plt.plot(x, y, '*r')

    def scatter(self, points_x: list, points_y: list, label:str) -> None:
        plt.plot(points_x, points_y, '.', label=label)

    def clear(self) -> None:
        self.__ax.clear()

def plotDotPaths(paths, smooth_paths, seg_idx):
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.size'] = 12

    front_path, middle_path, rear_path = paths
    front_smooth_path, middle_smooth_path, rear_smooth_path = smooth_paths
    front_seg_idx, middle_seg_idx, rear_seg_idx = seg_idx

    front_seg_points = [front_path[i] for i in front_seg_idx]
    middle_seg_points = [middle_path[i] for i in middle_seg_idx]
    rear_seg_points = [rear_path[i] for i in rear_seg_idx]

    # Create subplots
    fig, axs = plt.subplots(1, 3, figsize=(16, 8))

    # Plot rear_path_points
    axs[0].plot(*zip(*rear_path), 'k--', label='Original')
    axs[0].plot(*zip(*rear_smooth_path), '-', color='#DC5956', label='Smoothed', lw=2)
    axs[0].plot(*zip(*rear_seg_points), '.', color='#DC5956', label='Segments endpoints', markersize=10)
    # axs[0].plot(rear_smooth[:,0], rear_smooth[:,1], 'r-', label='Rear Path Points smooth')
    axs[0].set_title('Rear Path')
    axs[0].set_xlabel('X')
    axs[0].set_ylabel('Y')
    axs[0].axis('equal')
    axs[0].grid()
    axs[0].legend() 

    # Plot front_path_points
    axs[1].plot(*zip(*front_path), 'k--', label='Original')
    axs[1].plot(*zip(*front_smooth_path), '-', color='#358185', label='Smoothed', lw=2)
    axs[1].plot(*zip(*front_seg_points), '.', color='#358185', label='Segments endpoints', markersize=10)
    # axs[1].plot(front_smooth[:,0], front_smooth[:,1], 'g-', label='Front Path Points smooth')
    axs[1].set_title('Front Path')
    axs[1].set_xlabel('X')
    axs[1].set_ylabel('Y')
    axs[1].axis('equal')
    axs[1].grid()
    axs[1].legend()

    # Plot middle_path_points
    axs[2].plot(*zip(*middle_path), 'k--', label='Original')
    axs[2].plot(*zip(*middle_smooth_path), '-', color='#C27330', label='Smoothed', lw=2)
    axs[2].plot(*zip(*middle_seg_points), '.', color='#C27330', label='Segments endpoints', markersize=10)
    # axs[2].plot(middle_smooth[:,0], middle_smooth[:,1], 'b-', label='Middle Path Points smooth')
    axs[2].set_title('Middle Path')
    axs[2].set_xlabel('X')
    axs[2].set_ylabel('Y')
    axs[2].axis('equal')
    axs[2].grid()
    axs[2].legend()

    plt.tight_layout()
    plt.savefig('paths_smooth.pdf', format='pdf', dpi=150, bbox_inches='tight')
    plt.show()

def arc(config: list, seg=1) -> tuple[np.ndarray, np.ndarray, float]:
        k = config[2+seg]
        l = np.linspace(0, gv.L_VSS, 50)
        flag = -1 if seg == 1 else 1
        theta_array = config[2] + flag * k * l

        if abs(k) < 1e-6:
            x = np.array([0, flag * gv.L_VSS * np.cos(config[2])])
            y = np.array([0, flag * gv.L_VSS * np.sin(config[2])])
        else:
            x = np.sin(theta_array) / k - np.sin(config[2]) / k
            y = -np.cos(theta_array) / k + np.cos(config[2]) / k

        x += config[0]
        y += config[1]
        theta_end = func.normalizeAngle(theta_array[-1])
            
        return x, y, theta_end

def plotPath(traversed_path: list, rear_pos: list, traversed_line, plot_components) -> None:
    traversed_path[0].append(rear_pos[0])
    traversed_path[1].append(rear_pos[1])

    traversed_line.set_data(traversed_path[0], traversed_path[1])
    plot_components.append(traversed_line)
            
def plotPoints(points: list, components: tuple, plot_components) -> None:
    for point, component in zip(points, components):
        component.set_data([point[0]], [point[1]])
        plot_components.append(component)
    
def plotOrientation(middle_pos: list, orientation: float, orientation_line, plot_components) -> None:
    l = 0.05

    orientation_line.set_data([middle_pos[0], middle_pos[0] + l * np.cos(orientation)], 
                              [middle_pos[1], middle_pos[1] + l * np.sin(orientation)])
    plot_components.append(orientation_line)

def getConnEnds(x: float, y: float, theta_end: float, seg=1) -> tuple[np.ndarray, np.ndarray]:
    sign = -1 if seg == 1 else 1

    conn_start = np.array([x, y])
    conn_vec = gv.L_CONN * np.array([sign * np.cos(theta_end), sign * np.sin(theta_end)])
    conn_end = conn_start + conn_vec

    return conn_start, conn_end

def getLUCenter(corner: np.ndarray, lu_theta: float, seg=1) -> np.ndarray:
    sign = -1 if seg == 1 else 1

    lu_center = corner + gv.LU_SIDE/2 * np.array([
        sign * np.cos(lu_theta) + np.sin(lu_theta),  # x shift
        sign * np.sin(lu_theta) - np.cos(lu_theta)   # y shift
        ])
    
    return lu_center

def getLUCorners(corner: np.ndarray, lu_theta: float, seg=1) -> np.ndarray:
    lu_center = getLUCenter(corner, lu_theta, seg)

    lu_corners = lu_center + gv.LU_SIDE/2 * np.array([
        [-np.cos(lu_theta) - np.sin(lu_theta), -np.sin(lu_theta) + np.cos(lu_theta)],
        [-np.cos(lu_theta) + np.sin(lu_theta), -np.sin(lu_theta) - np.cos(lu_theta)],
        [np.cos(lu_theta) + np.sin(lu_theta), np.sin(lu_theta) - np.cos(lu_theta)],
        [np.cos(lu_theta) - np.sin(lu_theta), np.sin(lu_theta) + np.cos(lu_theta)],
        [-np.cos(lu_theta) - np.sin(lu_theta), -np.sin(lu_theta) + np.cos(lu_theta)]  # Close the square
        ])
    
    return lu_corners

def plotRobot(q: list, plot_components: list, components: tuple) -> None:
    vss1_line, vss2_line, conn1_line, conn2_line, lu1_square, lu2_square = components
    
    # Plot VSS
    x_vss1, y_vss1, theta_vss1_end = arc(q, seg=1)
    vss1_line.set_data(x_vss1, y_vss1)
    plot_components.append(vss1_line)

    x_vss2, y_vss2, theta_vss2_end = arc(q, seg=2)
    vss2_line.set_data(x_vss2, y_vss2)
    plot_components.append(vss2_line)

    # Plot connection lines
    # Front connection
    conn1_start, conn1_end = getConnEnds(x_vss1[-1], y_vss1[-1], theta_vss1_end)
    conn1_line.set_data([conn1_start[0], conn1_end[0]], [conn1_start[1], conn1_end[1]])
    plot_components.append(conn1_line)
    
    # Rear connection
    conn2_start, conn2_end = getConnEnds(x_vss2[-1], y_vss2[-1], theta_vss2_end, 2)
    conn2_line.set_data([conn2_start[0], conn2_end[0]], [conn2_start[1], conn2_end[1]])
    plot_components.append(conn2_line)

    # Plot LU squares
    # Front LU
    lu1_corners = getLUCorners(conn1_end, theta_vss1_end)
    lu1_square.set_data(lu1_corners[:, 0], lu1_corners[:, 1])
    plot_components.append(lu1_square)

    # Rear LU - connected at left top corner
    lu2_corners = getLUCorners(conn2_end, theta_vss2_end, 2)
    lu2_square.set_data(lu2_corners[:, 0], lu2_corners[:, 1])
    plot_components.append(lu2_square)