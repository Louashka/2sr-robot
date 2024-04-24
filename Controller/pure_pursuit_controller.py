import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/lytaura/Documents/PolyU/Research/2SR/Version 1/Multi agent/Control/2sr-swarm-control')
from Model import manipulandum as mp, global_var as gv, robot2sr
from grasping_controller import Grasp as grasp, Force
import path
import numpy as np
import random as rnd
from datetime import datetime
from typing import List
import math
from scipy.optimize import fsolve

GOAL_RADIUS = 0.05
lw = 2
alpha = 0.364748
c = 58.87274 * 10**(-3)

def getTangent(obj: mp.Shape, s: float):
    dx = 0
    dy = 0

    for h in range(obj.m):
        c = 2 * (h + 1) * np.pi
        arg = c * s
        exp = [-c * np.sin(arg),  c * np.cos(arg)]

        coef = obj.coeffs[h,:]
        dx += coef[0] * exp[0] + coef[1] * exp[1]
        dy += coef[2] * exp[0] + coef[3] * exp[1]

    theta = np.arctan(dy/dx)

    return theta

def getDirec(obj: mp.Shape, s: float):
    theta = getTangent(obj, s)

    p1 = obj.getPoint(s)
    p2 = [p1[0] + np.cos(theta), p1[1] + np.sin(theta)]
    p3 = [p1[0] + np.cos(theta - np.pi/2), p1[1] + np.sin(theta - np.pi/2)]

    cross_prod1 = (p1[0] - p2[0]) * (obj.y - p2[1]) - (p1[1] - p2[1]) * (obj.x - p2[0])
    cross_prod2 = (p1[0] - p2[0]) * (p3[1] - p2[1]) - (p1[1] - p2[1]) * (p3[0] - p2[0])

    condition = (abs(cross_prod1 * cross_prod2) - cross_prod1 * cross_prod2) / (-2 * cross_prod1 * cross_prod2)
    theta += condition * np.pi

    return theta

def getPoint(s, m, coeffs):
    x = 0
    y = 0

    for h in range(1, m+1):
        arg = 2 * h * s * np.pi
        exp = [np.cos(arg), np.sin(arg)]

        coef = coeffs[h-1,:]
        x += coef[0] * exp[0] + coef[1] * exp[1]
        y += coef[2] * exp[0] + coef[3] * exp[1]

    return [x, y]

def getTangent(s, m, coeffs):
    dx = 0
    dy = 0

    for h in range(1, m+1):
        c = 2 * h * np.pi
        arg = c * s
        exp = [-c * np.sin(arg),  c * np.cos(arg)]

        coef = coeffs[h-1,:]
        dx += coef[0] * exp[0] + coef[1] * exp[1]
        dy += coef[2] * exp[0] + coef[3] * exp[1]

    theta = np.arctan(dy/dx)

    return theta

def getDirec(s, m, coeffs):
    theta = getTangent(s, m, coeffs)

    p1 = getPoint(s, m, coeffs)
    p2 = [p1[0] + np.cos(theta), p1[1] + np.sin(theta)]
    p3 = [p1[0] + np.cos(theta + np.pi/2), p1[1] + np.sin(theta + np.pi/2)]

    cross_prod1 = (p1[0] - p2[0]) * (- p2[1]) - (p1[1] - p2[1]) * (- p2[0])
    cross_prod2 = (p1[0] - p2[0]) * (p3[1] - p2[1]) - (p1[1] - p2[1]) * (p3[0] - p2[0])

    condition = (abs(cross_prod1 * cross_prod2) - cross_prod1 * cross_prod2) / (-2 * cross_prod1 * cross_prod2)
    theta += condition * np.pi

    return theta

def contactKinematics(obj: mp.Shape, s: list, force_coef: list) -> np.ndarray:
    
    B_c_i = np.array([[1, 0], [0, 1], [0, 0]])
    G_list = []
    
    for s_i in s:
        cp = obj.getPoint(s_i)
        plt.plot(cp[0], cp[1], '*r', markersize = 16)

        cp_body_frame = getPoint(s_i, obj.m, obj.coeffs)
        cp_theta = (getDirec(s_i, obj.m, obj.coeffs))

        R_c_i = np.array([[np.cos(cp_theta), np.cos(cp_theta + np.pi / 2)],
                        [np.sin(cp_theta), np.sin(cp_theta + np.pi / 2)]])
        
        T_bc_i = np.block([[R_c_i, np.zeros((2, 1))], 
                         [np.array([-cp_body_frame[1], cp_body_frame[0]]).T.dot(R_c_i), 1]])
        
        G_i = T_bc_i.dot(B_c_i)
        G_list.append(G_i)        
        

        # plt.plot(cp_body_frame[0], cp_body_frame[1], '*m', markersize = 16)
        # plt.arrow(cp_body_frame[0], cp_body_frame[1], 0.05 * np.cos(cp_theta), 0.05 * np.sin(cp_theta), width=0.005)

    G = np.block(G_list)
    
    nu = 0.2 # friction coefficient
    hat_theta = np.arctan(nu) # semi-aperture angle of the friction cone

    # Rotation matrices of the force unit vectors w.r.t. {C_i}
    hat_f_c_y_1 = np.array([[-round(np.sin(-hat_theta), 3), round(np.cos(-hat_theta), 3)]]).T
    hat_f_c_y_2 = np.array([[-round(np.sin(hat_theta), 3), round(np.cos(hat_theta), 3)]]).T

    # A inear combination of unit vectors
    hat_F_c_i = np.block([[hat_f_c_y_1, hat_f_c_y_2]])
    # The matrix encoding all the unit vectors representing the boundary of the friction cones
    hat_F_c = np.kron(np.eye(len(s),dtype=int), hat_F_c_i)

    F_c = hat_F_c.dot(force_coef)

    # R_0 = np.array([[np.cos(obj.theta), -np.sin(obj.theta), 0], 
    #                 [np.sin(obj.theta), np.cos(obj.theta), 0], 
    #                 [0, 0, 1]])
    
    # q_dot = R_0.dot(G).dot(F_c)

    F_o = G.dot(F_c)

    return F_o

class PI:
    def __init__(self, kp=8.0, ki=0.1):
        """
        Define a PID controller class
        :param kp: float, kp coeff
        :param ki: float, ki coeff
        :param kd: float, kd coeff
        """
        self.kp = kp
        self.ki = ki
        self.total_error = 0.0

    def control(self, error):
        """
        PID main function, given an input, this function will output a control unit
        :param error: float, error term
        :return: float, output control
        """

        self.total_error += error * dt
        p_term = self.kp * error
        i_term = self.ki * self.total_error * dt
        output = p_term + i_term
        
        return output
    
class PID:
    def __init__(self):
        """
        Define a PID controller class
        :param kp: float, kp coeff
        :param ki: float, ki coeff
        :param kd: float, kd coeff
        """
        self.kp = [3, 3, 5]
        self.ki = [0.1] * 3
        self.kd = [0] * 3

        self.I = [0] * 3

        self.e_prev = [0] * 3
        self.time_prev = datetime.now()

    def control(self, e: np.ndarray) -> tuple[np.ndarray, float]:
        """
        PID main function, given an input, this function will output a control unit
        :param error: float, error term
        :return: float, output control
        """

        current_time = datetime.now()
        dt = (current_time - self.time_prev).total_seconds() 
        
        P = np.diag(self.kp).dot(e)
        self.I += np.diag(self.ki).dot(e) * dt
        D = np.diag(self.kd).dot(e - self.e_prev) / dt
        
        output = P + self.I + D

        self.e_prev = e
        self.time_prev = current_time
        
        return output, dt
    
class PD:
    def __init__(self):
        """
        Define a PID controller class
        :param kp: float, kp coeff
        :param ki: float, ki coeff
        :param kd: float, kd coeff
        """
        self.kp = [3.0, 3.0, 5.0]
        self.kd = [5.0] * 3

    def control(self, e_pos: np.ndarray, e_vel: np.ndarray) -> np.ndarray:
        
        P = np.diag(self.kp).dot(e_pos)
        D = np.diag(self.kd).dot(e_vel)
        
        output = P + D
        
        return output
    
def arc(theta0, k, seg=1) -> List[np.ndarray]:
    l = np.linspace(0, gv.L_VSS, 50)
    flag = -1 if seg == 1 else 1
    theta_array = theta0 + flag * k * l

    if k == 0:
        x = np.array([0, flag * gv.L_VSS * np.cos(theta0)])
        y = np.array([0, flag * gv.L_VSS * np.sin(theta0)])
    else:
        x = np.sin(theta_array) / k - np.sin(theta0) / k
        y = -np.cos(theta_array) / k + np.cos(theta0) / k

    theta = theta_array[-1]
        
    return [x, y, theta % (2 * np.pi)]

    
def plotRobot(config) -> None:

    # Plot the reference point
    plt.plot(config[0], config[1], 'b.', markersize=14)

    # Plot VS segments
    vss1 = arc(config[2], config[3])
    plt.plot(config[0] + vss1[0], config[1] + vss1[1], 'b-', lw=lw)

    vss2 = arc(config[2], config[4], 2)
    plt.plot(config[0] + vss2[0], config[1] + vss2[1], 'b-', lw=lw)

    # Plot VSS connectores
    vss1_conn_x = [config[0] + vss1[0][-1] - gv.L_CONN * np.cos(vss1[2]), config[0] + vss1[0][-1]]
    vss1_conn_y = [config[1] + vss1[1][-1] - gv.L_CONN * np.sin(vss1[2]), config[1] + vss1[1][-1]]
    plt.plot(vss1_conn_x, vss1_conn_y, 'k-', lw=lw)

    vss2_conn_x = [config[0] + vss2[0][-1], config[0] + vss2[0][-1] + gv.L_CONN * np.cos(vss2[2])]
    vss2_conn_y = [config[1] + vss2[1][-1], config[1] + vss2[1][-1] + gv.L_CONN * np.sin(vss2[2])]
    plt.plot(vss2_conn_x, vss2_conn_y, 'k-', lw=lw)

    # Plot the origins of locomotion units
    lu_head_x = vss1_conn_x[0] + np.sqrt(2) / 2 * gv.LU_SIDE * np.cos(vss1[2] + np.pi + np.pi / 4)
    lu_head_y = vss1_conn_y[0] + np.sqrt(2) / 2 * gv.LU_SIDE * np.sin(vss1[2] + np.pi + np.pi / 4)
    plt.plot(lu_head_x, lu_head_y, 'r*', markersize=10)

    lu_tail_x = vss2_conn_x[1] + np.sqrt(2) / 2 * gv.LU_SIDE * np.cos(vss2[2] - np.pi / 4)
    lu_tail_y = vss2_conn_y[1] + np.sqrt(2) / 2 * gv.LU_SIDE * np.sin(vss2[2] - np.pi / 4)
    plt.plot(lu_tail_x, lu_tail_y, 'r*', markersize=10)

    plt.plot([lu_head_x, lu_tail_x], [lu_head_y, lu_tail_y], 'k--')

    # Plot a body frame
    robot_x = (lu_head_x + lu_tail_x) / 2
    robot_y = (lu_head_y + lu_tail_y) / 2

    dy = lu_tail_y - lu_head_y
    dx = lu_tail_x - lu_head_x

    robot_theta = np.arctan(dy/dx)
    if dx < 0:
        robot_theta -= np.pi
    
    plt.plot(robot_x, robot_y, '*r', markersize=14)
    plt.plot([robot_x, robot_x + 0.05 * np.cos(robot_theta)], [robot_y, robot_y + 0.05 * np.sin(robot_theta)], '-r', lw='2')

    # Plot locomotion units
    LU_outline = np.array(
        [
            [-gv.LU_SIDE/2, gv.LU_SIDE/2, gv.LU_SIDE/2, -gv.LU_SIDE/2, -gv.LU_SIDE/2],
            [gv.LU_SIDE, gv.LU_SIDE, 0, 0, gv.LU_SIDE],
        ]
    )

    rot1 = np.array([[math.cos(vss1[2]), math.sin(vss1[2])], [-math.sin(vss1[2]), math.cos(vss1[2])]])
    rot2 = np.array([[math.cos(vss2[2]), math.sin(vss2[2])], [-math.sin(vss2[2]), math.cos(vss2[2])]])

    LU1_outline = (LU_outline.T.dot(rot1)).T
    LU2_outline = (LU_outline.T.dot(rot2)).T

    LU1_outline[0, :] += vss1_conn_x[0] + gv.LU_SIDE * (np.sin(vss1[2]) - np.cos(vss1[2]) / 2)
    LU1_outline[1, :] += vss1_conn_y[0] - gv.LU_SIDE * (np.cos(vss1[2]) + np.sin(vss1[2]) / 2)

    LU2_outline[0, :] += vss2_conn_x[-1] + gv.LU_SIDE * (np.sin(vss2[2]) + np.cos(vss2[2]) / 2)
    LU2_outline[1, :] += vss2_conn_y[-1] - gv.LU_SIDE * (np.cos(vss2[2]) - np.sin(vss2[2]) / 2)

    plt.plot(np.array(LU1_outline[0, :]).flatten(), np.array(LU1_outline[1, :]).flatten(), '-k', lw=lw)
    plt.plot(np.array(LU2_outline[0, :]).flatten(), np.array(LU2_outline[1, :]).flatten(), '-k', lw=lw)

def func(var, *args):
    k = var[0]
    c, l = args

    theta = k * l

    eq = c * theta - 2 * l * np.sin(theta / 2)

    return eq

if __name__ == "__main__":
    # Define the manipulandum shape
    heart_df = pd.read_csv('./Data/heart_contour.csv')[['distance', 'phase']].dropna()
    heart_r = heart_df['distance'].tolist()
    heart_theta = heart_df['phase'].tolist()
    heart = mp.Shape(11, [0.4, 0.32, 0.0 * np.pi], [heart_r, heart_theta])

    # Generate a random trajectory 
    trajectory = path.Trajectory(heart.position)
    target_pos = trajectory.targetPoint(heart.position, GOAL_RADIUS)
    target_pos.append(heart.theta)

    # Initial orientation of the manipulandum
    gamma = np.arctan2(target_pos[1] - heart.y, target_pos[0] - heart.x)

    # Real trajectory
    traj_shape_x = []
    traj_shape_y = []

    # Determine contact points
    s = [0] * 3

    grasp_model = grasp(heart, target_pos)
    solution_status = grasp_model.solve()

    if solution_status:
        results, s, force, q_new = grasp_model.parseResults()  

    # Constants
    force_unit_vectors = np.array([[0.196, -0.196], [0.981, 0.981]])
    VEL_COEF = 5
    mass = 0.2
    dt = 0.05
    
    # Target linear veloicites
    target_lin_vel = 0.15
    target_vel_x = target_lin_vel * np.cos(gamma)
    target_vel_y = target_lin_vel * np.sin(gamma)

    # Define a PD controller and a force optimization solver
    pd_controller = PD()
    force_model = Force(heart, s)

    # Initialise a robot
    robot2sr = robot2sr.Robot(1, [0.75, 0, 0])

    # Execute the object's manipulation
    
    while path.distance(heart.position, trajectory.goal) > GOAL_RADIUS:
        
        cp = []
        for s_i in s:
            cp.append(heart.getPoint(s_i))

        robot_target_theta = heart.getTangent(s[1]) + heart.theta

        c1 = path.distance(cp[1], cp[2])
        data1 = (c1, gv.L_VSS)
        k1_solution = fsolve(func, 15, args=data1)
        robot_target_k1 = k1_solution[0]
        print(robot_target_k1)

        c2 = path.distance(cp[0], cp[1])
        data2 = (c2, gv.L_VSS)
        k2_solution = fsolve(func, 0, args=data2)
        robot_target_k2 = k2_solution[0]
        print(robot_target_k1)

        robot2sr.k = [robot_target_k1, robot_target_k2]

        robot_target = [cp[1][0], cp[1][1], robot_target_theta, 0, 0]

        robot_dist = path.distance(cp[1], robot2sr.position)
        # print(robot_dist)
        if robot_dist > 0.01:
            q_tilda = (np.array(robot_target) - robot2sr.config) * dt
            v_robot = 40 * np.matmul(np.linalg.pinv(robot2sr.jacobain), q_tilda)
            robot2sr.update(v_robot, dt)
        else:
            # Find next target pose
            target_pos = trajectory.targetPoint(heart.position, GOAL_RADIUS)
            theta_traj = np.arctan2(target_pos[1] - heart.y, target_pos[0] - heart.x)
            target_theta = theta_traj - gamma
            
            if abs(heart.theta - target_theta) > np.pi:
                if target_theta > 0:
                    target_theta -= 2 * np.pi
                else:
                    target_theta += 2 * np.pi 

            target_pos.append(target_theta)

            # Calculate position and velocity errors
            pos_error = np.array(target_pos) - heart.pose
            vel_error = np.array([target_vel_x, target_vel_y, VEL_COEF * (target_theta - heart.theta)]) - heart.velocity

            # Target acceleration
            acc = pd_controller.control(pos_error, vel_error)
            heart.update(acc) # !!Comment and replace with real-time tracking

            # Target net force acting on the manipulandum 
            tau = mass * acc
            print('Target force: ' + np.array2string(tau))

            force_model.update(tau.tolist())
            solution_status = force_model.solve()

            if solution_status:
                results, force, F_o = force_model.parseResults()  

            print('Supposed force: [' + ','.join(map(str, F_o)) + ']')
            # print('Force coeffs: [' + ','.join(map(str, force)) + ']')

            traj_shape_x.append(heart.x)
            traj_shape_y.append(heart.y)

        # result_F_o = contactKinematics(heart, s, force)
        # print('Result force: ' + np.array2string(result_F_o))
        # print('')

        # Determine the robot's velocities
        # q_robot_dot = ...
        
        # robot2sr.update(q_robot_dot + [0, 0])
        # print(robot2sr.config)

        i = 0
        plt.cla()
        for s_i in s:
            cp = heart.getPoint(s_i)

            force_coeffs = 10 * np.array([force[i], force[i+1]]).T 
            f_c_i = force_unit_vectors.dot(force_coeffs)

            theta_wc_i = heart.theta + getDirec(s_i, heart.m, heart.coeffs)
            T_wc_i = np.array([[np.cos(theta_wc_i), -np.sin(theta_wc_i), cp[0]],
                            [np.sin(theta_wc_i), np.cos(theta_wc_i), cp[1]],
                            [0, 0, 1]])
            
            f_wc_i = T_wc_i.dot(np.append(f_c_i, 1))

            plt.plot(cp[0], cp[1], '*r', markersize = 12)
            # plt.plot([cp[0], f_wc_i[0]], [cp[1], f_wc_i[1]], '-r')

            # cp_theta_bf = getDirec(s_i, heart.m, heart.coeffs)
            # F_o_list[0] += 0.196 * (force[i] - force[i+1]) * np.cos(cp_theta_bf) - 0.981 * (force[i] + force[i+1]) * np.sin(cp_theta_bf)
                                                                                    
            i += 2

        # print('Calc force: [' + ','.join(map(str, F_o_list)) + ']')
        # print('')

        plt.plot(heart.x, heart.y, 'or', markersize = 12)
        plt.plot(heart.parametric_contour[0], heart.parametric_contour[1], '-b')
        plt.plot(trajectory.traj_x, trajectory.traj_y, "--k", linewidth=3, label="course")
        plt.plot(traj_shape_x, traj_shape_y, "-r", linewidth=3, label="trajectory")
        plotRobot(robot2sr.config)

        plt.axis("equal")
        plt.pause(0.05)    

    # Execute grasping by the robot


