from Controller import path
from Model import global_var as gv
import numpy as np
import tkinter as tk
import tkinter.font as font
import matplotlib.pyplot as plt
from typing import List
import math

GOAL_RADIUS = 0.05
dt = 0.05

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

def plotRobot(config) -> None:
    
    # Plot a body frame
    plt.plot(config[0], config[1], '*r', markersize=14)
    plt.plot([config[0], config[0] + 0.05 * np.cos(config[2])], [config[1], config[1] + 0.05 * np.sin(config[2])], '-r', lw='2')

    # Plot VS segments
    vss1 = arc(config[2], config[3])
    plt.plot(config[0] + vss1[0], config[1] + vss1[1], '-b', lw='5')

    vss2 = arc(config[2], config[4], 2)
    plt.plot(config[0] + vss2[0], config[1] + vss2[1], '-b', lw='5')

    # Plot VSS connectores
    vss1_conn_x = [config[0] + vss1[0][-1], config[0] + vss1[0][-1] + gv.L_CONN * np.cos(vss1[2])]
    vss1_conn_y = [config[1] + vss1[1][-1], config[1] + vss1[1][-1] + gv.L_CONN * np.sin(vss1[2])]
    plt.plot(vss1_conn_x, vss1_conn_y, '-k', lw='5')

    vss2_conn_x = [config[0] + vss2[0][-1], config[0] + vss2[0][-1] + gv.L_CONN * np.cos(vss2[2])]
    vss2_conn_y = [config[1] + vss2[1][-1], config[1] + vss2[1][-1] + gv.L_CONN * np.sin(vss2[2])]
    plt.plot(vss2_conn_x, vss2_conn_y, '-k', lw='5')

    # Plot locomotion units
    LU_outline = np.array(
        [
            [-gv.LU_SIDE/2, gv.LU_SIDE/2, gv.LU_SIDE/2, -gv.LU_SIDE/2, -gv.LU_SIDE/2],
            [gv.LU_SIDE/2, gv.LU_SIDE/2, -gv.LU_SIDE/2, -gv.LU_SIDE/2, gv.LU_SIDE/2],
        ]
    )

    rot1 = np.array([[math.cos(vss1[2]), math.sin(vss1[2])], [-math.sin(vss1[2]), math.cos(vss1[2])]])
    rot2 = np.array([[math.cos(vss2[2]), math.sin(vss2[2])], [-math.sin(vss2[2]), math.cos(vss2[2])]])

    LU1_outline = (LU_outline.T.dot(rot1)).T
    LU2_outline = (LU_outline.T.dot(rot2)).T

    LU1_outline[0, :] += vss1_conn_x[-1] + gv.LU_SIDE * np.cos(vss1[2]) / 2
    LU1_outline[1, :] += vss1_conn_y[-1] + gv.LU_SIDE * np.sin(vss1[2]) / 2

    LU2_outline[0, :] += vss2_conn_x[-1] + gv.LU_SIDE * np.cos(vss2[2]) / 2
    LU2_outline[1, :] += vss2_conn_y[-1] + gv.LU_SIDE * np.sin(vss2[2]) / 2

    plt.plot(np.array(LU1_outline[0, :]).flatten(), np.array(LU1_outline[1, :]).flatten(), '-k')
    plt.plot(np.array(LU2_outline[0, :]).flatten(), np.array(LU2_outline[1, :]).flatten(), '-k')

def arc(theta0, k, seg=1) -> List[np.ndarray]:
    l = np.linspace(0, gv.L_VSS, 50)
    theta_array = theta0 + k * l

    if k == 0:
        x = np.array([0, gv.L_VSS * np.cos(theta0)])
        y = np.array([0, gv.L_VSS * np.sin(theta0)])
    else:
        x = np.sin(theta_array) / k - np.sin(theta0) / k
        y = -np.cos(theta_array) / k + np.cos(theta0) / k

    theta = theta_array[-1]

    if seg == 1:
        x = -x
        y = -y
        theta += -np.pi
        
    return [x, y, theta % (2 * np.pi)]

class Robot:
    def __init__(self, x: float, y: float, theta: float, k: List[float]=[0.0, 0.0], s: List[int]=[0, 0]):
        """
        Define a robot class
        :param x: float, x position
        :param y: float, y position
        :param theta: float, robot orientation
        :param k: list, list of VSF curvature values 
        :param vel: list, list of velocity control inputs
        """
        self.x = x
        self.y = y
        self.theta = theta
        self.k = k
        self.s = s

    @property
    def position(self) -> List[float]:
        return [self.x, self.y]
    
    @property
    def pose(self) -> List[float]:
        return self.position + [self.theta]
    
    @property
    def config(self) -> np.ndarray:
        return np.array(self.pose + self.k)
    
    @config.setter
    def config(self, value) -> None:
        self.x = value[0]
        self.y = value[1]
        self.theta = value[2]
        self.k[0] = value[3]
        self.k[1] = value[4]
    
    def update(self, vel: np.ndarray) -> None:
        q_dot = np.dot(self.jacobain, vel)
        self.config += q_dot * dt

    @property
    def jacobain(self) -> np.ndarray:
        # RIGID STATE

        flag_rigid = int(not (self.s[0] or self.s[1]))  # checks if both
        # segments are rigid
        J_rigid = flag_rigid * np.array([[np.cos(self.theta), -np.sin(self.theta), 0],
                                        [np.sin(self.theta), np.cos(self.theta), 0],
                                        [0, 0, 1],
                                        [0, 0, 0],
                                        [0, 0, 0]])

        # SOFT STATE

        # Central angles of VSS arcs
        alpha = np.array(self.k) * gv.L_VSS
        # Angles of 3 types of logarithmic spirals
        theta = np.divide(np.tile(np.reshape(alpha, (2, 1)), 3), gv.M) - np.pi

        # Spiral constants
        a = gv.SPIRAL_COEF[0]
        b = np.tile(gv.SPIRAL_COEF[1], (2, 1))
        for i in range(2):
            if self.k[i] > 0:
                b[i] = -b[i]
                theta[i] += 2 * np.pi

        rho = a * np.exp(b * theta)  # spiral radii

        # Proportionality coefficient of the rate of VSS curvature change
        Kappa = np.divide(gv.M, gv.L_VSS * rho)
        # Proportionality coefficient of the rate of 2SRR orientation change
        Phi = np.divide(gv.M, rho)

        flag_soft1 = int(not self.s[1] and self.s[0])  # checks if VSS1 is soft
        flag_soft2 = int(not self.s[0] and self.s[1])  # checks if VSS2 is soft
        flag_full_soft = int(self.s[0] and self.s[1])  # checks if both VSSs are soft

        #  Proportionality coefficients of the rate of change of 2SRR position coordinates
        Delta = np.zeros((2, 2))
        Delta[:, 0] = (flag_soft2 * Kappa[1, 1] + flag_full_soft * Kappa[1, 2]) * \
            self.__positionDerivative(rho[1, 0], self.theta, self.k[1], self.k[1], 2).ravel()
        Delta[:, 1] = (flag_soft1 * Kappa[0, 1] + flag_full_soft * Kappa[0, 2]) * \
            self.__positionDerivative(rho[0, 0], self.theta, self.k[0], self.k[0], 1).ravel()

        J_soft = np.array([[-flag_soft2 * Phi[1, 1] - flag_full_soft * Phi[1, 2],
                            flag_soft1 * Phi[0, 1] + flag_full_soft * Phi[0, 2]],
                        [flag_soft1 * Kappa[0, 0] + flag_full_soft * Kappa[0, 2],
                            flag_soft1 * Kappa[0, 1] + flag_full_soft * Kappa[0, 2]],
                        [flag_soft2 * Kappa[1, 1] + flag_full_soft * Kappa[1, 2],
                            flag_soft2 * Kappa[1, 0] + flag_full_soft * Kappa[1, 2]]])

        J_soft = np.concatenate((Delta, J_soft), axis=0)

        # complete hybrid Jacobian matrix
        J = np.concatenate((J_soft, J_rigid), axis=1)

        return J
    
    def __positionDerivative(self, r: float, phi: float, k: float, k0: float, seg: int) -> np.ndarray:
        if seg == 1:
            pos = np.array([[-0.0266 * r * np.sin(phi + 0.0266 * k - 0.04 * k0) - 0.0006 * np.sin(phi + 0.04 * k - 0.04 * k0)],
                            [0.0266 * r * np.cos(phi + 0.0266 * k - 0.04 * k0) + 0.0006 * np.cos(phi + 0.04 * k - 0.04 * k0)]])
            # pos = bodyFramePosition(-1)
        elif seg == 2:
            pos = np.array([[-0.0266 * r * np.sin(phi - 0.0266 * k + 0.04 * k0) - 0.0006 * np.sin(phi - 0.04 * k + 0.04 * k0)],
                            [0.0266 * r * np.cos(phi - 0.0266 * k + 0.04 * k0) + 0.0006 * np.cos(phi - 0.04 * k + 0.04 * k0)]])
            
        return pos
    

class Trajectory:
    def __init__(self, traj_x: List[float], traj_y: List[float]):
        """
        Define a trajectory class
        :param traj_x: list, list of x position
        :param traj_y: list, list of y position
        """
        self.traj_x = traj_x
        self.traj_y = traj_y
        self.last_idx = 0

    def point(self, idx):
        return [self.traj_x[idx], self.traj_y[idx]]
    
    def tangent(self, i):
        # Calculate derivatives using central differences method
        if i == 0:
            dx = self.traj_x[i+1] - self.traj_x[i]
            dy = self.traj_y[i+1] - self.traj_y[i]
        elif i == len(self.traj_x) - 1:
            dx = self.traj_x[i] - self.traj_x[i-1]
            dy = self.traj_y[i] - self.traj_y[i-1]
        else:
            dx = 0.5 * (self.traj_x[i+1] - self.traj_x[i-1])
            dy = 0.5 * (self.traj_y[i+1] - self.traj_y[i-1])

        # Calculate tangent angle
        tangent_angle = np.arctan2(dy, dx)

        return tangent_angle

    def targetConfig(self, pos: List[float]):
        """
        Get the next look ahead point
        :param pos: list, robot position
        :return: list, target point
        """
        target_idx = self.last_idx
        target_point = self.point(target_idx)
        curr_dist = distance(pos, target_point)

        while curr_dist < GOAL_RADIUS and target_idx < len(self.traj_x) - 1:
            target_idx += 1
            target_point = self.point(target_idx)
            curr_dist = distance(pos, target_point)

        self.last_idx = target_idx
        config = self.point(target_idx) + [self.tangent(target_idx) + np.pi/2] + [0, 0]

        return config
    
class PI:
    def __init__(self, kp=10.0, ki=0.0):
        """
        Define a PID controller class
        :param kp: float, kp coeff
        :param ki: float, ki coeff
        """
        self.kp = kp
        self.ki = ki
        self.total_error = np.array([0.0] * 5)

    def feeback(self, error: np.ndarray) -> np.ndarray:
        """
        PI main function, given an input, this function will output a control unit
        :param error: float, error term
        :return: float, output control
        """
        self.total_error += error
        p_term = self.kp * error
        i_term = self.ki * self.total_error * dt
        output = p_term + i_term

        return output


if __name__ == "__main__":
    robot = Robot(0, 0, 0)

    # target course
    # traj_x = np.arange(0, 4, 0.02)
    # traj_y = [0.75 * x * math.sin(2.5 * x) for x in traj_x]
    traj = path.generateCurve()
    traj_x = traj[:, 0]
    traj_y = traj[:, 1]
    traj = Trajectory(traj_x, traj_y)
    goal = traj.point(len(traj_x) - 1)

    # create PI controller
    PI_acc = PI()

    # real trajectory
    traj_robot_x = []
    traj_robot_y = []

    # target velocity
    VEL_COEF = 2
    vel = [0] * 5

    sim_time = 10
    timer = 0
    time = 0
    times = []

    target_config = traj.targetConfig(robot.position)
    vel = [0] * 5
    amplitude = []


    # plt.figure(figsize=(12, 8))

    while distance(robot.position, goal) > GOAL_RADIUS:

        if timer > sim_time:
            target_config = traj.targetConfig(robot.position)
            timer = 0

        q_tilda = (target_config - robot.config) * VEL_COEF
        target_vel = np.matmul(np.linalg.pinv(robot.jacobain), q_tilda)

        times.append(time)
        time += 1
        amplitude.append(vel[2] / target_vel[2])

        vel_err = target_vel - vel
        acc = PI_acc.feeback(vel_err)
        vel = target_vel + acc * dt

        robot.update(vel)
        timer += 1

        print(timer)
        print(target_vel[2])

        # store the trajectory
        # traj_robot_x.append(robot.x)
        # traj_robot_y.append(robot.y)

        # plots
        plt.cla()
        # plt.plot(traj_x, traj_y, "-k", linewidth=3, label="course")
        # plt.plot(traj_robot_x, traj_robot_y, "-r", linewidth=2, label="trajectory")
        # plt.plot(target_config[0], target_config[1], "og", ms=5, label="target point")
        # plotRobot(robot.config)
        # plt.xlabel("x[m]")
        # plt.ylabel("y[m]")
        plt.plot(times, amplitude, '-k')
        plt.axis("equal")
        # plt.legend()
        plt.grid(True)
        plt.pause(0.5)

