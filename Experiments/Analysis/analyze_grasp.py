import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd



file_path = 'Experiments/Data/Tracking/Grasp/grasp_exp_circle_2025-03-02_20-24-09.json'

with open(file_path, 'r') as f:
    file_data = json.load(f)
    approach_data = file_data['approach']['1']['tracking']

robot_states = approach_data['robot']['states']
robot_vel = approach_data['robot']['target_vel']

robot_states = np.array(robot_states)
robot_vel = np.array(robot_vel)

fig, axs = plt.subplots(2, 3, figsize=(15, 8))

# Plot robot states
axs[0, 0].plot(robot_states[:, 0], label='X Position')
axs[0, 0].set_title('Robot States - X Position')
axs[0, 0].set_xlabel('Time')
axs[0, 0].set_ylabel('X Position')
axs[0, 0].legend()

axs[0, 1].plot(robot_states[:, 1], label='Y Position', color='orange')
axs[0, 1].set_title('Robot States - Y Position')
axs[0, 1].set_xlabel('Time')
axs[0, 1].set_ylabel('Y Position')
axs[0, 1].legend()

axs[0, 2].plot(robot_states[:, 2], label='Z Position', color='green')
axs[0, 2].set_title('Robot States - Z Position')
axs[0, 2].set_xlabel('Time')
axs[0, 2].set_ylabel('Z Position')
axs[0, 2].legend()

# Plot robot velocities
axs[1, 0].plot(robot_vel[:, 0], label='X Velocity', color='red')
axs[1, 0].set_title('Robot Velocities - X Velocity')
axs[1, 0].set_xlabel('Time')
axs[1, 0].set_ylabel('X Velocity')
axs[1, 0].legend()

axs[1, 1].plot(robot_vel[:, 1], label='Y Velocity', color='purple')
axs[1, 1].set_title('Robot Velocities - Y Velocity')
axs[1, 1].set_xlabel('Time')
axs[1, 1].set_ylabel('Y Velocity')
axs[1, 1].legend()

axs[1, 2].plot(robot_vel[:, 2], label='Z Velocity', color='brown')
axs[1, 2].set_title('Robot Velocities - Z Velocity')
axs[1, 2].set_xlabel('Time')
axs[1, 2].set_ylabel('Z Velocity')
axs[1, 2].legend()

plt.tight_layout()
plt.show()





