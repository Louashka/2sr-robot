import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/lytaura/Documents/PolyU/Research/2SR/Version 1/Multi agent/Control/2sr-swarm-control')
from Model import manipulandum as mp, global_var as gv
from grasping_controller import Grasp as grasp
import path
import numpy as np
import random as rnd

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

def contactKinematics():
    s_dist = (gv.L_VSS + gv.L_CONN + gv.LU_SIDE / 2) / heart.perimeter
    
    s = [0] * 3
    s[0] = rnd.random()
    for i in range(1, 3):
        s[i] = s[i-1] + s_dist
        if s[i] >= 1:
            s[i] -= 1

    B_c_i = np.array([[1, 0], [0, 1], [0, 0]])
    G_list = []
    
    for s_i in s:
        cp = heart.getPoint(s_i)
        plt.plot(cp[0], cp[1], '*r', markersize = 16)

        cp_body_frame = getPoint(s_i, heart.m, heart.coeffs)
        cp_theta = (getDirec(s_i, heart.m, heart.coeffs))

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

    force_coef = np.array([1] * 6).T

    F_c = hat_F_c.dot(force_coef)

    R_0 = np.array([[np.cos(heart.theta), -np.sin(heart.theta), 0], 
                    [np.sin(heart.theta), np.cos(heart.theta), 0], 
                    [0, 0, 1]])
    
    q_dot = R_0.dot(G).dot(F_c)
    print(q_dot)

    pose_new = heart.pose + q_dot * 0.01
    plt.plot(pose_new[0], pose_new[1], 'or', markersize=16)


if __name__ == "__main__":
    # Define the manipulandum shape
    heart_df = pd.read_csv('./Data/heart_contour.csv')[['distance', 'phase']].dropna()
    heart_r = heart_df['distance'].tolist()
    heart_theta = heart_df['phase'].tolist()
    heart = mp.Shape(11, [0.4, 0.32, 0.8 * np.pi], [heart_r, heart_theta])

    plt.plot(heart.contour[0], heart.contour[1], '.k')
    # plt.plot(heart.default_contour[0], heart.default_contour[1], '--b')
    plt.plot(heart.x, heart.y, 'ok', markersize=16)

    # Extrapolate manipulandums' contour
    contour_approx = []
    for s in np.linspace(0, 1):
        pos_target = heart.getPoint(s)
        contour_approx.append(pos_target)

    contour_approx = np.array(contour_approx)
    plt.plot(contour_approx[:, 0], contour_approx[:, 1], '-b')

    # Generate a random trajectory 
    # trajectory = path.Trajectory(heart.position)
    arrow_l = 0.15
    direction  = rnd.random() * 2 * np.pi
    # plt.arrow(heart.x, heart.y, arrow_l * np.cos(direction), arrow_l * np.sin(direction), width=0.01, color='red')

    # Determine contact points
    # x_c, y_c = f(s) 
    coef = 0.02
    q_target = [heart.x + coef * np.cos(direction), heart.y + coef * np.sin(direction), heart.theta]
    # print('Current q: ' + str(heart.pose))
    print('Target q: ' + str(q_target))
    plt.plot(q_target[0], q_target[1], 'or', markersize=16)

    # contactKinematics()


    grasp_model = grasp(heart, q_target)
    solution_status = grasp_model.solve()

    force_unit_vectors = np.array([[0.196, -0.196], [0.981, 0.981]])

    if solution_status:
        results, s, force, q_new = grasp_model.parseResults()  
        print('New q: ' + str(q_new))
        plt.plot(q_new[0], q_new[1], 'ob', markersize=10)

        i = 0
        for s_i in s:
            cp = heart.getPoint(s_i)

            force_coeffs = np.array([force[i], force[i+1]]).T 
            f_c_i = force_unit_vectors.dot(force_coeffs)
            print(f_c_i)

            i += 2

            theta_wc_i = heart.theta + getDirec(s_i, heart.m, heart.coeffs)
            T_wc_i = np.array([[np.cos(theta_wc_i), -np.sin(theta_wc_i), cp[0]],
                               [np.sin(theta_wc_i), np.cos(theta_wc_i), cp[1]],
                               [0, 0, 1]])
            
            f_wc_i = T_wc_i.dot(np.append(f_c_i, 1))

            plt.plot(cp[0], cp[1], '*r', markersize = 16)
            plt.plot([cp[0], f_wc_i[0]], [cp[1], f_wc_i[1]], '-r')
            # plt.arrow(cp[0], cp[1], 0.05 * np.cos(direc), 0.05 * np.sin(direc), width=0.005)
    

    # Execute grasping by the robot
    

    # plt.plot(trajectory.traj_x, trajectory.traj_y, '--k')
    plt.axis('equal')
    plt.show()

