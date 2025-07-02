import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import matplotlib.pyplot as plt
from entities import global_var as gv, robot_state
import plotlib


robot = robot_state.Model(1, 1, 1, np.pi/4, k1=0, k2=15, stiffness=[0, 1])
rp = plotlib.RobotPlot()

fig, ax = plt.subplots()
rp.plot_robot(ax, robot)

plt.axis('equal')
plt.show()



