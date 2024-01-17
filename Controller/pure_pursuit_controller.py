import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/lytaura/Documents/PolyU/Research/2SR/Version 1/Multi agent/Control/2sr-swarm-control')
from Model import manipulandum as mp, global_var as gv
from grasping_controller import Grasp as grasp
import path
import numpy as np
import random as rnd
from datetime import datetime

GOAL_RADIUS = 0.05

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

    R_0 = np.array([[np.cos(obj.theta), -np.sin(obj.theta), 0], 
                    [np.sin(obj.theta), np.cos(obj.theta), 0], 
                    [0, 0, 1]])
    
    q_dot = R_0.dot(G).dot(F_c)

    return q_dot

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

    # real trajectory
    traj_shape_x = []
    traj_shape_y = []

    # Determine contact points
    force_unit_vectors = np.array([[0.196, -0.196], [0.981, 0.981]])
    # VEL_COEF = 3
    dt = 0.05

    # force_coef = np.array([1] * 6).T

    # s_dist = (gv.L_VSS + gv.L_CONN + gv.LU_SIDE / 2) / heart.perimeter
    
    # s = [0] * 3
    # s[0] = rnd.random()
    # for i in range(1, 3):
    #     s[i] = s[i-1] + s_dist
    #     if s[i] >= 1:
    #         s[i] -= 1

    # t = 0
    

    # while t < 10:
    #     heart.pose += contactKinematics(heart, s, force_coef) * dt
    #     print(heart.theta)

    #     traj_shape_x.append(heart.x)
    #     traj_shape_y.append(heart.y)

    #     t += dt

    #     plt.cla()

    #     i = 0
    #     for s_i in s:
    #         cp = heart.getPoint(s_i)
    #         f_c_i = force_unit_vectors.dot(np.array([force_coef[i], force_coef[i+1]]).T)

    #         i += 2

    #         theta_wc_i = heart.theta + getDirec(s_i, heart.m, heart.coeffs)
    #         T_wc_i = np.array([[np.cos(theta_wc_i), -np.sin(theta_wc_i), cp[0]],
    #                         [np.sin(theta_wc_i), np.cos(theta_wc_i), cp[1]],
    #                         [0, 0, 1]])
            
    #         f_wc_i = T_wc_i.dot(np.append(f_c_i, 1))

    #         plt.plot(cp[0], cp[1], '*r', markersize = 12)
    #         plt.plot([cp[0], f_wc_i[0]], [cp[1], f_wc_i[1]], '-r')
    #     plt.plot(heart.x, heart.y, 'or', markersize = 12)
    #     plt.plot(heart.parametric_contour[0], heart.parametric_contour[1], '-b')
    #     plt.plot(traj_shape_x, traj_shape_y, "--k", linewidth=3)

    #     plt.axis("equal")
    #     plt.pause(dt)    

    s = [0] * 3

    grasp_model = grasp(heart, target_pos)
    solution_status = grasp_model.solve()

    if solution_status:
        results, s, force, q_new = grasp_model.parseResults()  

    gamma = np.arctan2(target_pos[1] - heart.y, target_pos[0] - heart.x)

    PI_lin_acc = PI()
    PI_ang_acc = PI()

    pid = PID()

    target_lin_vel = 0.15
    VEL_COEF = 5
    
    while path.distance(heart.position, trajectory.goal) > GOAL_RADIUS:
        # store the trajectory
        theta_traj = np.arctan2(target_pos[1] - heart.y, target_pos[0] - heart.x)

        target_pos = trajectory.targetPoint(heart.position, GOAL_RADIUS)
        target_theta = theta_traj - gamma
        
        if abs(heart.theta - target_theta) > np.pi:
            if target_theta > 0:
                target_theta -= 2 * np.pi
            else:
                target_theta += 2 * np.pi 

        target_pos.append(target_theta)
        # print(target_pos[-1])
        
        # q_error = np.array(target_pos) - heart.pose
        # q_tilda = VEL_COEF * q_error
        # target_vel = np.matmul(np.linalg.pinv(heart.rotation_matrix), q_tilda)

        # q_dot = q_tilda
        # heart.x += q_dot[0] * dt
        # heart.y += q_dot[1] * dt
        # heart.theta += q_dot[2] * dt

        # use PI to control the vehicle
        # lin_vel_err = target_lin_vel - heart.lin_vel
        # lin_acc = PI_lin_acc.control(lin_vel_err)

        # theta_traj = np.arctan2(target_pos[1] - heart.y, target_pos[0] - heart.x)
        # ang_vel_err = VEL_COEF * (target_theta - heart.theta) - heart.ang_vel
        # ang_acc = PI_ang_acc.control(ang_vel_err)

        # heart.update(gamma, lin_acc, ang_acc)

        pos_error = np.array(target_pos) - heart.pose
        # vel_error = np.array([target_lin_vel, VEL_COEF * (target_theta - heart.theta)]) - heart.velocity

        q_dot, dt = pid.control(pos_error)
        print('Acc: ' + np.array2string(q_dot) + ', dt = ' + str(dt))
        heart.update(q_dot, dt)

        traj_shape_x.append(heart.x)
        traj_shape_y.append(heart.y)

        # i = 0
        plt.cla()
        for s_i in s:
            cp = heart.getPoint(s_i)

            # force_coeffs = np.array([force[i], force[i+1]]).T 
            # f_c_i = force_unit_vectors.dot(force_coeffs)

            # i += 2

            # theta_wc_i = heart.theta + getDirec(s_i, heart.m, heart.coeffs)
            # T_wc_i = np.array([[np.cos(theta_wc_i), -np.sin(theta_wc_i), cp[0]],
            #                 [np.sin(theta_wc_i), np.cos(theta_wc_i), cp[1]],
            #                 [0, 0, 1]])
            
            # f_wc_i = T_wc_i.dot(np.append(f_c_i, 1))

            plt.plot(cp[0], cp[1], '*r', markersize = 12)
            # plt.plot([cp[0], f_wc_i[0]], [cp[1], f_wc_i[1]], '-r')

        plt.plot(heart.x, heart.y, 'or', markersize = 12)
        plt.plot(heart.parametric_contour[0], heart.parametric_contour[1], '-b')
        plt.plot(trajectory.traj_x, trajectory.traj_y, "--k", linewidth=3, label="course")
        plt.plot(traj_shape_x, traj_shape_y, "-r", linewidth=3, label="trajectory")

        plt.axis("equal")
        plt.pause(0.05)    

    # Execute grasping by the robot


