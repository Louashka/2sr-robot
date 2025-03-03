import os
import json
import cv2
import numpy as np
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from shapely import affinity
import seaborn as sns
import pandas as pd

agent_width = 0.07
agent_length = 0.32

colors = ['#dd8f07', '#d06261', 'k', '#0173b2']

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['text.usetex'] = False
plt.rcParams['font.size'] = 14

marker_size = 50

obstacles_file_path = f'Experiments/Data/Tracking/Grasp/obstacles.json'

def readTraverseFiles():
    directory = 'Experiments/Data/Tracking/Grasp'
    traverse_rm_files = [f for f in os.listdir(directory) if f.startswith('traverse_rm_') and f.endswith('.json')]
    
    traverse_data = []

    for file in traverse_rm_files:
        file_path = os.path.join(directory, file)
        with open(file_path, 'r') as f:
            file_data = json.load(f)
            traverse_data.append(file_data['traversing'])

    return traverse_data

def getObstacles():
    all_experiments = []

    with open(obstacles_file_path, 'r') as f:
        obstacles_file_data = json.load(f)

        rm_experiments = obstacles_file_data['rm']
        for exp in rm_experiments.values():
            obstacles_polygons = []
            
            for obstacle_corners in exp:
                obstacles_polygons.append(Polygon(obstacle_corners))
            
            all_experiments.append(obstacles_polygons)

    return all_experiments

def getRobotPolygon(pose):
    """Create robot polygon at given pose (x, y, theta)"""
    # Create rectangle centered at origin
    l, w = agent_length, agent_width
    points = [
        (-l/2, -w/2),
        (l/2, -w/2),
        (l/2, w/2),
        (-l/2, w/2)
    ]
    # Create polygon and transform it
    robot = Polygon(points)
    # Rotate and translate
    robot = affinity.rotate(robot, pose[2] * 180/np.pi)
    robot = affinity.translate(robot, pose[0], pose[1])
    return robot

def processData():
    fig, axs = plt.subplots(2, 4, figsize=(16,4.3))

    k1_all = []
    k2_all = []

    for idx in range(4):        
        # Get data for this experiment
        data = all_traverse_data[idx]

        target_x = []
        target_y = []
        target_th = []
        target_timestamps = []

        robot_x = []
        robot_y = []
        robot_th = []
        robot_k1 = []
        robot_k2 = []
        robot_timestamps = []

        vel_x = []
        vel_y = []
        vel_omega = []

        obstacles = all_obstacles[idx]

        for data_item in list(data.values())[:-1]:
            timestamps = data_item['tracking']['time']
            target_config = data_item['target_config']
            robot_states = data_item['tracking']['robot_states']
            robot_vel = data_item['tracking']['target_vel']

            target_x.append(target_config[0])
            target_y.append(target_config[1])
            target_th.append(target_config[2])

            target_timestamps.append(timestamps[-1])

            robot_x.extend([state[0] for state in robot_states])
            robot_y.extend([state[1] for state in robot_states])
            robot_th.extend([state[2] for state in robot_states])
            robot_k1.extend([state[3] for state in robot_states])
            robot_k2.extend([state[4] for state in robot_states])
            robot_timestamps.extend(timestamps)

            vel_x.extend([100 * vel[0] for vel in robot_vel])
            vel_y.extend([100 * vel[1] for vel in robot_vel])
            vel_omega.extend([100 * vel[2] for vel in robot_vel])


        k1_all.append(robot_k1)
        k2_all.append(robot_k2)
        
        clearance = []
        
        for x, y, th in zip(robot_x, robot_y, robot_th):
            robot_polygon = getRobotPolygon((x, y, th))
            min_distance = float('inf')
            for obstacle in obstacles:
                distance = robot_polygon.distance(obstacle)
                min_distance = min(min_distance, distance)
            clearance.append(100 * min_distance)


        robot_x = 10 * np.array(robot_x)
        target_x = 10 * np.array(target_x)

        robot_y = 10 * np.array(robot_y)
        target_y = 10 * np.array(target_y)
        
        axs[0,0].plot(robot_timestamps, robot_x, color=colors[idx])
        axs[0,0].scatter(target_timestamps, target_x, color=colors[idx], marker='*', s=marker_size)

        if idx == 2:
            axs[0,1].plot(robot_timestamps, robot_y, color=colors[idx], label='meas')
            axs[0,1].scatter(target_timestamps, target_y, color=colors[idx], 
                             marker='*', s=30, label='target')
        else:
            axs[0,1].plot(robot_timestamps, robot_y, color=colors[idx])
            axs[0,1].scatter(target_timestamps, target_y, color=colors[idx], marker='*', s=marker_size)
        
        axs[0,2].plot(robot_timestamps, robot_th, color=colors[idx])
        axs[0,2].scatter(target_timestamps, target_th, color=colors[idx], marker='*', s=marker_size)

        axs[0,3].plot(robot_timestamps, clearance, color=colors[idx], label=f'{idx+4}')
        
        # Apply median filter to velocity data
        window_size = 10 
        vel_x = np.array(vel_x)
        vel_y = np.array(vel_y)
        vel_x_filtered = pd.Series(vel_x).rolling(window=window_size, center=True).median()
        vel_y_filtered = pd.Series(vel_y).rolling(window=window_size, center=True).median()

        if idx == 2:            
            axs[1,0].plot(robot_timestamps, vel_x_filtered, color=colors[idx], label=f'{idx+4}')
            axs[1,1].plot(robot_timestamps, vel_y_filtered, color=colors[idx], label=f'{idx+4}')
            axs[1,2].plot(robot_timestamps, vel_omega, color=colors[idx], label=f'{idx+4}')


    axs[0,0].set_ylabel('x [dm]')
    axs[0,1].set_ylabel('y [dm]')
    
    axs[0,2].set_ylabel(r'$\theta$ [rad]')
    axs[0,3].set_ylabel('Clearance [cm]')

    axs[1,0].set_xlabel('Time [s]')
    axs[1,0].set_ylabel(r'$v_x$ [cm/s]')

    axs[1,1].set_xlabel('Time [s]')
    axs[1,1].set_ylabel(r'$v_y$ [cm/s]')

    axs[1,2].set_xlabel('Time [s]')
    axs[1,2].set_ylabel(r'$\omega$ [rad/s]')

    axs[0,1].legend(prop={'size': 12})
    axs[0,3].legend(ncol=2, prop={'size': 12})
    axs[1,0].legend(prop={'size': 12})

    sns.set_theme(palette="pastel")
    
    curvature_df = pd.DataFrame({
        'exp trial': np.concatenate([np.repeat(['4', '5', '6', '7'], [len(k) for k in k1_all]),
                                np.repeat(['4', '5', '6', '7'], [len(k) for k in k2_all])]),
        r'curvature [$m^{-1}$]': [i for s in k1_all for i in s] + [i for s in k2_all for i in s],
        'seg': ['VSS1'] * sum(len(k) for k in k1_all) + ['VSS2'] * sum(len(k) for k in k2_all)
    })

    # Draw a nested boxplot to show bills by day and time
    sns.boxplot(x="exp trial", y=r'curvature [$m^{-1}$]',
                hue="seg", palette=["r", "b"],
                data=curvature_df, ax=axs[1,3])
    
    axs[1,3].legend(loc='upper right', prop={'size': 9}, bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.savefig('rm_trav_analysis.pdf', format='pdf', dpi=150, bbox_inches='tight', transparent=True)
    # plt.show()
#

all_traverse_data = readTraverseFiles()
all_obstacles = getObstacles()
processData()

