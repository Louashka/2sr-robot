import sys
sys.path.append('D:/Robot 2SR/2sr-swarm-control')
import os
import glob
import json
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

def analyse_data(path, data):
    timestamps = []
    object_poses = []
    object_target_velocities = [] # body velocities (2d)
    robot_poses = []
    robot_target_velocities = [] # body velocities (2d)

    for data_point in data:
        timestamps.append(data_point['time'])

        object = data_point['object']
        object_poses.append(object['pose'])
        object_target_velocities.append(object['target_velocity'])

        robot = data_point['robot']
        robot_poses.append(robot['pose'])
        robot_target_velocities.append(robot['target_velocity'])

    # Calculate distances and errors
    distances_to_path = []
    yaw_errors = []
    object_orientations = []
    robot_orientations = []

    for i in range(len(data)):
        object_pose = data[i]['object']['pose']
        robot_pose = data[i]['robot']['pose']
        
        # Calculate distance from object to path (using nearest point logic)
        distances = cdist([object_pose[:2]], target_path[:, :2])[0]
        distances_to_path.append(np.min(distances))
        
        # Calculate yaw errors
        object_yaw = object_pose[2]
        path_yaw = target_path[np.argmin(distances), 2]
        
        yaw_errors.append(np.abs(object_yaw - path_yaw))
        object_orientations.append(object_yaw)

    # Prepare velocities
    object_target_velocities = np.array(object_target_velocities)
    robot_target_velocities = np.array(robot_target_velocities)

    # Calculate object and robot velocities from their poses
    object_actual_velocities = np.diff(object_poses, axis=0) / np.diff(np.array(timestamps)[:, np.newaxis], axis=0)
    robot_actual_velocities = np.diff(robot_poses, axis=0) / np.diff(np.array(timestamps)[:, np.newaxis], axis=0)

    # Plotting
    fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

    # 1.1 Distance to path and yaw errors
    axs[0].plot(timestamps, distances_to_path, label='Distance to Path', color='blue')
    axs[0].plot(timestamps, yaw_errors, label='Object Yaw Error', color='orange')
    axs[0].set_title('Distance to Path and Yaw Errors')
    axs[0].set_ylabel('Distance / Yaw Error')
    axs[0].legend()
    axs[0].grid(True)
    

    # 1.2 Object body velocities
    axs[1].plot(timestamps, object_target_velocities[:, 0], label='Object Target Velocity X', color='blue')
    # axs[1].plot(timestamps[1:], object_actual_velocities[:, 0], label='Object Recorded Velocity X', linestyle='--', color='cyan')
    axs[1].plot(timestamps, object_target_velocities[:, 1], label='Object Target Velocity Y', color='orange')
    # axs[1].plot(timestamps[1:], object_actual_velocities[:, 1], label='Object Recorded Velocity Y', linestyle='--', color='lightcoral')
    axs[1].plot(timestamps, object_target_velocities[:, 2], label='Object Target Angular Velocity', color='green')
    # axs[1].plot(timestamps[1:], object_actual_velocities[:, 2], label='Object Recorded Angular Velocity', linestyle='--', color='lightgreen')
    axs[1].set_title('Object Body Velocities')
    axs[1].set_ylabel('Velocity (m/s)')
    axs[1].legend()
    axs[1].grid(True)

    # 1.3 Robot body velocities
    axs[2].plot(timestamps, robot_target_velocities[:, 0], label='Robot Target Velocity X', color='blue')
    # axs[2].plot(timestamps[1:], robot_actual_velocities[:, 0], label='Robot Recorded Velocity X', linestyle='--', color='cyan')
    axs[2].plot(timestamps, robot_target_velocities[:, 1], label='Robot Target Velocity Y', color='orange')
    # axs[2].plot(timestamps[1:], robot_actual_velocities[:, 1], label='Robot Recorded Velocity Y', linestyle='--', color='lightcoral')
    axs[2].plot(timestamps, robot_target_velocities[:, 2], label='Robot Target Angular Velocity', color='green')
    # axs[2].plot(timestamps[1:], robot_actual_velocities[:, 2], label='Robot Recorded Angular Velocity', linestyle='--', color='lightgreen')
    axs[2].set_title('Robot Body Velocities')
    axs[2].set_ylabel('Velocity (m/s)')
    axs[2].legend()
    axs[2].grid(True)

    plt.xlabel('Timestamps (s)')
    plt.tight_layout()
    plt.show()

    # Plotting second set
    fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

    # 2.1 First object vs robot linear velocity
    axs[0].plot(timestamps, object_target_velocities[:, 0], label='Object Target', color='blue')
    # axs[0].plot(timestamps[1:], object_actual_velocities[:, 0], label='Object Recorded', linestyle='--', color='cyan')
    axs[0].plot(timestamps, robot_target_velocities[:, 1], label='Robot Target', color='orange')
    # axs[0].plot(timestamps[1:], robot_actual_velocities[:, 0], label='Robot Recorded', linestyle='--', color='lightcoral')
    axs[0].set_title('Object vs Robot Linear Velocity X')
    axs[0].set_ylabel('Linear Velocity (m/s)')
    axs[0].legend()
    axs[0].grid(True)

    # 2.2 Second object vs robot linear velocity
    axs[1].plot(timestamps, object_target_velocities[:, 1], label='Object Target', color='blue')
    # axs[1].plot(timestamps[1:], object_actual_velocities[:, 1], label='Object Recorded', linestyle='--', color='cyan')
    axs[1].plot(timestamps, robot_target_velocities[:, 0], label='Robot Target', color='orange')
    # axs[1].plot(timestamps[1:], robot_actual_velocities[:, 1], label='Robot Recorded', linestyle='--', color='lightcoral')
    axs[1].set_title('Object vs Robot Linear Velocity Y')
    axs[1].set_ylabel('Linear Velocity (m/s)')
    axs[1].legend()
    axs[1].grid(True)

    axs[2].plot(timestamps, object_target_velocities[:, 2], label='Object Target', color='blue')
    # axs[2].plot(timestamps[1:], object_actual_velocities[:, 2], label='Object Recorded', linestyle='--', color='cyan')
    axs[2].plot(timestamps, robot_target_velocities[:, 2], label='Robot Target', color='orange')
    # axs[2].plot(timestamps[1:], robot_actual_velocities[:, 2], label='Robot Recorded', linestyle='--', color='lightcoral')
    axs[2].set_title('Object vs Robot Angular Velocity')
    axs[2].set_ylabel('Linear Velocity (m/s)')
    axs[2].legend()
    axs[2].grid(True)

    plt.xlabel('Timestamps (s)')
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
    json_dir = os.path.join(parent_dir, 'Data/Tracking/Object_transport/')

    # json_file_name = os.path.join(json_dir, 'cheescake_1.json')
    json_file_name = os.path.join(json_dir, 'ellipse_2024-10-30_13-59-23.json')

    # json_file_name = glob.glob(os.path.join(json_dir, '*ellipse_2024*.json'))

    json_file = open(json_file_name)
    data = json.load(json_file)

    target_path = np.array([[p['x'], p['y'], p['yaw']] for p in data['path']])

    results = analyse_data(target_path, data['tracking'])