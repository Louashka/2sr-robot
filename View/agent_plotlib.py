import matplotlib.pyplot as plt
import numpy as np
from typing import List
from Model import agent, global_var as gv

def plotAgent(robot: agent.Robot) -> None:        

    # Plot VS segments
    vss1 = robot.arc()
    plt.plot(robot.x + vss1[0], robot.y + vss1[1], '-b', lw='5')

    vss2 = robot.arc(2)
    plt.plot(robot.x + vss2[0], robot.y + vss2[1], '-b', lw='5')

    # Plot VSS connectores
    vss1_conn_x = [robot.x + vss1[0][-1] - gv.L_CONN * np.cos(vss1[2]), robot.x + vss1[0][-1]]
    vss1_conn_y = [robot.y + vss1[1][-1] - gv.L_CONN * np.sin(vss1[2]), robot.y + vss1[1][-1]]
    plt.plot(vss1_conn_x, vss1_conn_y, '-k', lw='5')

    vss2_conn_x = [robot.x + vss2[0][-1], robot.x + vss2[0][-1] + gv.L_CONN * np.cos(vss2[2])]
    vss2_conn_y = [robot.y + vss2[1][-1], robot.y + vss2[1][-1] + gv.L_CONN * np.sin(vss2[2])]
    plt.plot(vss2_conn_x, vss2_conn_y, '-k', lw='5')

        # Plot a body frame
    plt.plot(robot.x, robot.y, '*r')
    # plt.arrow(robot.x, robot.y, 0.05 * np.cos(robot.theta), 0.05 * np.sin(robot.theta), width=0.005, color='red')
    plt.plot([robot.x, robot.x + 0.05 * np.cos(robot.theta)], 
                [robot.y, robot.y + 0.05 * np.sin(robot.theta)], '-r', lw='2')

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

    # Plot geometrical centers of the locomoton units
    lu_head = robot.lu_position(1)
    plt.plot(lu_head[0], lu_head[1], '*k')

    lu_tail = robot.lu_position(2)
    plt.plot(lu_tail[0], lu_tail[1], '*k')

    # Plot the line connecting LU's
    plt.plot([lu_head[0], lu_tail[0]], [lu_head[1], lu_tail[1]], '--k')

    # Plot the body frame
    robot_bf = robot.body_frame
    plt.plot(robot_bf[0], robot_bf[1], '*r', markersize=14)
    plt.plot([robot_bf[0], robot_bf[0] + 0.05 * np.cos(robot_bf[2])], 
             [robot_bf[1], robot_bf[1] + 0.05 * np.sin(robot_bf[2])], '-r', lw='2')
    

def animateAgent(robot_id: int, config_array: List[np.ndarray], dt: float) -> None:

    robot = agent.Robot(robot_id, 0, 0, 0, 0, 0)

    plt.pause(2)

    for i in range(len(config_array)):
        robot.config = config_array[i]

        config_x = np.array(config_array)[:i,0]
        config_y = np.array(config_array)[:i,1]

        plt.cla()
        plt.plot(config_x, config_y, 'm--')
        plotAgent(robot)

        plt.axis("equal")
        plt.ylim(bottom=-0.3, top=0.45)
        plt.xlim(left=-0.3, right=0.45)
        plt.grid(True)
        plt.pause(dt)

