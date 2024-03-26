import matplotlib.pyplot as plt
import numpy as np
import math
from Model import global_var as gv, agent
from typing import List

lw = 2
alpha = 0.364748
c = 58.87274 * 10**(-3)

def distance(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return np.hypot(dx, dy)

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

if __name__ == "__main__":
    ref_point = [0, 0, 0.0 * np.pi/4]
    robot2sr = agent.Robot(1, ref_point, [10, -15])

    # v = np.array([0, 0, 0, 0, 0]).T
    v = np.array([0, 0, 0, 0, 1.5]).T

    total_t = 3
    t = 0
    dt = 0.05

    config_array = []

    # vss1 = arc(robot2sr.ref_point[2], robot2sr.k[0])
    # rot1 = np.array([[math.cos(vss1[2]), math.sin(vss1[2])], [-math.sin(vss1[2]), math.cos(vss1[2])]])

    # p1_array = []
    # p1_x = robot2sr.ref_point[0] + vss1[0][-1] - (gv.L_CONN + gv.LU_SIDE/2) * np.cos(vss1[2])
    # p1_y = robot2sr.ref_point[1] + vss1[1][-1] - (gv.L_CONN + gv.LU_SIDE/2) * np.sin(vss1[2])

    # p1_array.append([p1_x, p1_y])

    # r = distance(p1_array[0], robot2sr.position)

    # dy = p1_y - robot2sr.y
    # dx = p1_x - robot2sr.x
    # phi = np.arctan(dy/dx)

    plotRobot(robot2sr.config)

    plt.axis("equal")
    plt.grid(True)
    # plt.xlim(-0.25, 0.2)
    # plt.ylim(-0.25, 0.2)
    plt.ylim(bottom=-0.3, top=0.25)
    plt.xlim(left=-0.3, right=0.25)
    plt.pause(10)

    while t < total_t:

        robot2sr.update(v, dt)
        config_array.append(robot2sr.config)

        config_x = np.array(config_array)[:,0]
        config_y = np.array(config_array)[:,1]

        # T = np.array([[1, 0, -r * np.cos(phi + robot2sr.theta)],
        #               [0, 1, -r * np.sin(phi + robot2sr.theta)],
        #               [0, 0, 1]])
        
        # p1_dot = T.dot(q_dot[2:])
        # p1_current = p1_array[-1]
        # p1_array.append([p1_current[0] + p1_dot[0] * dt, p1_current[1] + p1_dot[1] * dt])

        t += dt

        plt.cla()
        plt.plot(config_x, config_y, 'k--')
        plotRobot(robot2sr.config)
        # plt.plot(p1[0], p1[1], 'm*', markersize=14)
        # plt.plot([p1[0], p1[0] + 0.05 * np.cos(phi)], [p1[1], p1[1] + 0.05 * np.sin(phi)], 'r-')

        plt.axis("equal")
        # plt.xlim(-0.25, 0.2)
        # plt.ylim(-0.25, 0.2)
        plt.ylim(bottom=-0.3, top=0.25)
        plt.xlim(left=-0.3, right=0.25)
        plt.grid(True)
        plt.pause(dt)

    # plt.plot(config_x, config_y, 'k--')
    # plotRobot(config_array[0])

    # plt.axis("equal")
    # plt.grid(True)
    # plt.pause(10)

    # for config in config_array:

    #     plt.cla()
    #     plt.plot(config_x, config_y, 'k--')
    #     plotRobot(config)
    #     # plt.plot(p1[0], p1[1], 'm*', markersize=14)
    #     # plt.plot([p1[0], p1[0] + 0.05 * np.cos(phi)], [p1[1], p1[1] + 0.05 * np.sin(phi)], 'r-')

    #     plt.axis("equal")
    #     plt.grid(True)
    #     plt.pause(dt)

    plt.pause(20)

