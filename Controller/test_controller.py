import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
sys.path.append('/Users/lytaura/Documents/PolyU/Research/2SR/Version 1/Multi agent/Control/2sr-swarm-control')
from Model import manipulandum as mp, global_var as gv
import random as rnd

def getDirec(obj: mp.Shape, s: float):
    theta = obj.getTangent(s)

    p1 = obj.getPoint(s)
    p2 = [p1[0] + np.cos(theta), p1[1] + np.sin(theta)]
    p3 = [p1[0] + np.cos(theta - np.pi/2), p1[1] + np.sin(theta - np.pi/2)]

    cross_prod1 = (p1[0] - p2[0]) * (obj.y - p2[1]) - (p1[1] - p2[1]) * (obj.x - p2[0])
    cross_prod2 = (p1[0] - p2[0]) * (p3[1] - p2[1]) - (p1[1] - p2[1]) * (p3[0] - p2[0])

    condition = (abs(cross_prod1 * cross_prod2) - cross_prod1 * cross_prod2) / (-2 * cross_prod1 * cross_prod2)
    theta += condition * np.pi

    return theta

def distance(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return np.hypot(dx, dy)

if __name__ == "__main__":

    # plt.arrow(0, 0, 1, 0, color='black', width=0.02)
    # plt.arrow(0, 0, 0, 1, color='black', width=0.02)

    # f_unit1 = np.array([0.196, 0.981]).T
    # f_unit2 = np.array([-0.196, 0.981]).T

    # plt.arrow(0, 0, f_unit1[0], f_unit1[1], color='blue', width=0.02)
    # plt.arrow(0, 0, f_unit2[0], f_unit2[1], color='blue', width=0.02)

    # f_unit = np.column_stack((f_unit1, f_unit2))
    # f_coeffs = np.array([1.0, -0.5])
    # f = f_unit.dot(f_coeffs)

    # plt.arrow(0, 0, f[0], f[1], color='red', width=0.02)

    # Define the manipulandum shape
    heart_df = pd.read_csv('./Data/heart_contour.csv')[['distance', 'phase']].dropna()
    heart_r = heart_df['distance'].tolist()
    heart_theta = heart_df['phase'].tolist()
    heart = mp.Shape(11, [0.4, 0.32, 0.3 * np.pi], [heart_r, heart_theta])

    plt.plot(heart.parametric_contour[0], heart.parametric_contour[1], '-k')

    s_dist = (gv.L_VSS + gv.L_CONN + gv.LU_SIDE / 2) / heart.perimeter
    s = [0] * 3
    # s[0] = rnd.random()
    s[0] = 0.40
    for i in range(1, 3):
        s[i] = s[i-1] + s_dist
        if s[i] >= 1:
            s[i] -= 1

    points = []
    for s_i in s:
        cp = heart.getPoint(s_i)
        points.append(cp)
        plt.plot(cp[0], cp[1], '*r', markersize = 12)

    b_frame_x = points[1][0]
    b_frame_y = points[1][1]
    b_frame_theta = heart.theta + heart.getTangent(s[1])
    print(b_frame_theta)

    scale = 0.1
    b_frame_dx = scale * np.cos(b_frame_theta)
    b_frame_dy = scale * np.sin(b_frame_theta)

    plt.arrow(b_frame_x, b_frame_y, b_frame_dx, b_frame_dy, color='red', width=0.005)

    head_dist = distance(points[0], points[1])
    head_theta = (np.arctan((points[0][1] - points[1][1])/(points[0][0] - points[1][0])) + np.pi) % (2 * np.pi)

    # plt.plot([points[1][0], points[1][0] + head_dist * np.cos(head_theta)], 
    #          [points[1][1], points[1][1] + head_dist * np.sin(head_theta)], '-r')

    tail_dist = distance(points[1], points[2])
    tail_theta = (np.arctan((points[2][1] - points[1][1])/(points[2][0] - points[1][0])) + np.pi) % (2 * np.pi)

    # plt.plot([points[1][0], points[1][0] + tail_dist * np.cos(tail_theta)], 
    #          [points[1][1], points[1][1] + tail_dist * np.sin(tail_theta)], '-r')

    head_theta -= b_frame_theta
    tail_theta -= b_frame_theta

    q_robot = np.array([b_frame_x, b_frame_y, b_frame_theta])
    rot_robot = np.array([[np.cos(q_robot[2]), -np.sin(q_robot[2]), 0],
                    [np.sin(q_robot[2]), np.cos(q_robot[2]), 0],
                    [0, 0, 1]])
    v_robot = np.array([0.2, 0.1, -1])

    dt = 0.3

    q_robot_new = q_robot + rot_robot.dot(v_robot.T) * dt

    head_new = q_robot_new[:-1] + [head_dist * np.cos(q_robot_new[2] + head_theta), head_dist * np.sin(q_robot_new[2] + head_theta)]
    tail_new = q_robot_new[:-1] + [tail_dist * np.cos(q_robot_new[2] + tail_theta), tail_dist * np.sin(q_robot_new[2] + tail_theta)]

    plt.plot(q_robot_new[0], q_robot_new[1], '*m', markersize = 12)
    plt.plot(head_new[0], head_new[1], '*m', markersize = 12)
    plt.plot(tail_new[0], tail_new[1], '*m', markersize = 12)


    plt.axis('equal')
    plt.show()