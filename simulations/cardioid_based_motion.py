import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from entities import global_var as gv, robot_state
import plotlib

k_max = np.pi / gv.L_VSS

rp = plotlib.RobotPlot()

vss1 = rp.arc((0, 0, 0), 0)
center = (vss1[0][-1], vss1[1][-1], vss1[-1])

robot = robot_state.Model(1, *center, k1=0, k2=0, stiffness=[0, 1])
direction = 1

cardioid_no = 1

phi = np.linspace(0, 2 * np.pi, 50)
cardioid_x = -2 * gv.CARDIOID_A[cardioid_no-1] * (1 - np.cos(phi)) * np.cos(phi) - gv.CARDIOID_OFFSET[cardioid_no-1] + center[0]
cardioid_y = 2 * gv.CARDIOID_A[cardioid_no-1] * (1 - np.cos(phi)) * np.sin(phi) + center[1]

fig, ax = plt.subplots()

def update(frame):
    global direction
    
    ax.clear()

    ax.plot(cardioid_x, cardioid_y, '.k')
    rp.plot_robot(ax, robot)

    # Set fixed axis limits for both axes
    ax.set_aspect('equal')
    ax.set_xlim(-0.10, 0.26)
    ax.set_ylim(-0.15, 0.15)

    if abs(robot.k2) >= k_max:
        direction *= -1

    robot.k2 += direction

ani = animation.FuncAnimation(fig=fig, func=update, frames=162)

# Save the animation as a GIF
writer = animation.PillowWriter(fps=15)
ani.save('multimedia/cardioid1_animation.gif', writer=writer)

# plt.show()



