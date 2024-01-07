import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/lytaura/Documents/PolyU/Research/2SR/Version 1/Multi agent/Control/2sr-swarm-control')
from Model import manipulandum as mp
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
    theta += condition * np.pi + np.pi / 2

    return theta


if __name__ == "__main__":
    # Define the manipulandum shape
    heart_df = pd.read_csv('./Data/heart_contour.csv')[['distance', 'phase']].dropna()
    heart_r = heart_df['distance'].tolist()
    heart_theta = heart_df['phase'].tolist()
    heart = mp.Shape(11, [0.4, 0.32, 0.8 * np.pi], [heart_r, heart_theta])

    plt.plot(heart.contour[0], heart.contour[1], '.k')
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
    print('Current q: ' + str(heart.pose))
    print('Target q: ' + str(q_target))
    plt.plot(q_target[0], q_target[1], 'or', markersize=16)

    grasp_model = grasp(heart, q_target)
    solution_status = grasp_model.solve()

    if solution_status:
        results, s, force, q_new = grasp_model.parseResults()  
        print('New q: ' + str(q_new))
        plt.plot(q_new[0], q_new[1], 'om', markersize=16)

        for s_i in s:
            print('Contact point: ' + str(s_i))
            cp = heart.getPoint(s_i)
            direc = heart.theta + getDirec(heart, s_i)

            plt.plot(cp[0], cp[1], '*r', markersize = 16)
            plt.arrow(cp[0], cp[1], 0.05 * np.cos(direc), 0.05 * np.sin(direc), width=0.005)
    

    # Execute grasping by the robot
    

    # plt.plot(trajectory.traj_x, trajectory.traj_y, '--k')
    plt.axis('equal')
    plt.show()

