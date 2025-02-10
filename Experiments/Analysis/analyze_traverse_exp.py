import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

blue_color = '#0E595C'
red_color = '#DC5956'
dark_grey_color = '#333333'

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['text.usetex'] = False
plt.rcParams['font.size'] = 8

def read_rm_files():
    directory = 'Experiments/Data/Tracking/Grasp'
    traverse_rm_files = [f for f in os.listdir(directory) if f.startswith('traverse_rm_') and f.endswith('.json')]
    
    traverse_data = []
    transport_data = []

    for file in traverse_rm_files:
        file_path = os.path.join(directory, file)
        with open(file_path, 'r') as f:
            file_data = json.load(f)
            traverse_data.append(file_data['traversing'])
            transport_data.append(file_data['transport'])

    return traverse_data, transport_data

def process_traverse_data(all_traverse_data):
    # Create a 3x4 subplot layout
    fig = plt.figure(figsize=(14, 8))
    
    # Create gridspec with different row heights and custom spacing
    gs1 = fig.add_gridspec(1, 4, 
                         width_ratios=[1, 1, 1, 1],
                         wspace=0.16,
                         bottom=0.33)  # Different spacing between rows: [space between 1&2, space between 2&3]

    # Rest of the code remains the same...
    # First row - 3D trajectories
    gs2 = fig.add_gridspec(2, 4, 
                         width_ratios=[1, 1, 1, 1],
                         height_ratios=[0.5, 0.5],  # Make middle row shorter
                         top = 0.5,
                         wspace=0.16,
                         hspace=0.01)  # Different spacing between rows: [space between 1&2, space between 2&3]
    for idx in range(4):        
        # Get data for this experiment
        data = all_traverse_data[idx]

        target_x = []
        target_y = []
        target_timestamps = []

        robot_x = []
        robot_y = []
        robot_timestamps = []

        dist_errors = []

        theta_errors = []

        for data_item in list(data.values())[:-1]:
            timestamps = data_item['tracking']['time']
            target_config = data_item['target_config']
            robot_states = data_item['tracking']['robot_states']

            target_x.append(target_config[0])
            target_y.append(target_config[1])

            target_timestamps.append(timestamps[-1])
            
            robot_x.extend([state[0] for state in robot_states])
            robot_y.extend([state[1] for state in robot_states])
            robot_timestamps.extend(timestamps)

            dist_errors.extend([100 * np.sqrt((target_config[0]-state[0])**2 + 
                                        (target_config[1]-state[1])**2) for state in robot_states])
            
            theta_errors.extend([(state[2]-target_config[2]) for state in robot_states])

        # Plot robot trajectory
        ax = fig.add_subplot(gs1[0, idx], projection='3d')
        ax.plot(robot_x, robot_y, robot_timestamps, '-', color=dark_grey_color, label='Meas', linewidth=1)
        
        # Plot target point
        ax.scatter(target_x, target_y, target_timestamps, c=red_color, marker='*', s=35, label='Target')
        
        # Set labels only for leftmost plot (idx=0)
        if idx == 0:
            ax.zaxis.set_rotate_label(False)
            ax.set_zlabel('Time [s]', rotation=90)
            ax.legend(loc='center right')

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        # Adjust view angle
        ax.view_init(elev=7, azim=70)

        # Second row - X and Y errors
        ax_dist = fig.add_subplot(gs2[0, idx])
        
        # Make the plot more rectangular by adjusting aspect ratio
        ax_dist.set_box_aspect(0.5)  # This makes the plot more rectangular
            
        # Plot errors
        ax_dist.plot(robot_timestamps, dist_errors, '-', color=blue_color, linewidth=1)        
        # Set labels and ticks
        if idx == 0:
            ax_dist.set_ylabel('Dist to target [cm]')


        ax_theta_error = fig.add_subplot(gs2[1, idx])
        
        # Make the plot more rectangular by adjusting aspect ratio
        ax_theta_error.set_box_aspect(0.5)  # This makes the plot more rectangular

        # Plot errors
        ax_theta_error.plot(robot_timestamps, theta_errors, '-', color=blue_color, linewidth=1)        
        # Set labels and ticks
        ax_theta_error.set_xlabel('Time [s]')
        if idx == 0:
            ax_theta_error.set_ylabel('Theta error [rad]')    

    plt.savefig('rm_traverse_analysis.pdf', format='pdf', dpi=150, bbox_inches='tight')
    plt.show()

def process_grasp(all_traverse_data):
    fig = plt.figure(figsize=(14, 10))
    
    # Create gridspec with different row heights and custom spacing
    gs = fig.add_gridspec(5, 4, 
                         width_ratios=[1, 1, 1, 1],
                         wspace=0.18,
                         hspace=0.3)
    
    k1_heating_avrg = 0
    k1_cooling_avrg = 0
    k2_heating_avrg = 0
    k2_cooling_avrg = 0
    
    for idx in range(4):        
        # Get data for this experiment
        data = all_traverse_data[idx]
        grasp_data = list(data.values())[-1]

        timestamps = grasp_data['tracking']['time']
        target_config = grasp_data['target_config']
        robot_states = grasp_data['tracking']['robot_states']
        stiffness = grasp_data['tracking']['target_mm']
        temp_data = grasp_data['tracking']['temperature']

        temp_timestamps = []
        k1_temp = []
        k2_temp = []

        for i in range(len(temp_data)):
            if temp_data[i] is not None:
                rel_time = np.array(temp_data[i]['relative_timestamps']) + timestamps[i]
                temp_timestamps.extend(rel_time.tolist())

                k1_temp.extend([temp[0] for temp in temp_data[i]['meas']])
                k2_temp.extend([temp[1] for temp in temp_data[i]['meas']])

        k1_heating = True
        k1_cooling = True
        k2_heating = True
        k2_cooling = True
        
        for i in range(1, len(k1_temp)):
            if k1_heating:
                if k1_temp[i] >= 63 and k1_temp[i] > k1_temp[0]:
                    k1_heating = False
                    k1_heating_avrg += temp_timestamps[i]-temp_timestamps[0]
                    print(f'k1 heating time: {temp_timestamps[i]-temp_timestamps[0]}')
            if k2_heating:
                if k2_temp[i] >= 63 and k2_temp[i] > k2_temp[0]:
                    k2_heating = False
                    k2_heating_avrg += temp_timestamps[i]-temp_timestamps[0]
                    print(f'k2 heatin time: {temp_timestamps[i]-temp_timestamps[0]}')
            

            if k1_cooling:
                if k1_temp[i] <= 53 and k1_temp[i] < k1_temp[0]:
                    # k1_cooling = False
                    k1_cooling_avrg += temp_timestamps[i]-temp_timestamps[0]
                    print(f'k1 cooling time: {temp_timestamps[i]-temp_timestamps[0]}')
            if k2_cooling:
                if k2_temp[i] <= 53 and k2_temp[i] < k2_temp[0]:
                    # k2_cooling = False
                    k2_cooling_avrg += temp_timestamps[i]-temp_timestamps[0]
                    print(f'k2 cooling time: {temp_timestamps[i]-temp_timestamps[0]}')
        
        print()

        dist_errors = ([100 * np.sqrt((target_config[0]-state[0])**2 + 
                                     (target_config[1]-state[1])**2) for state in robot_states])
        theta_errors = [(state[2]-target_config[2]) for state in robot_states]
        k1_errors = [(state[3]-target_config[3]) for state in robot_states]
        k2_errors = [(state[4]-target_config[4]) for state in robot_states]
        k1_stiff = [stiff[0] for stiff in stiffness]
        k2_stiff = [stiff[1] for stiff in stiffness]

        ax_dist = fig.add_subplot(gs[0, idx])
        ax_dist.plot(timestamps, dist_errors, '-', color=dark_grey_color, linewidth=1)        
        # Set labels and ticks
        if idx == 0:
            ax_dist.set_ylabel('Dist to target [cm]')

        ax_theta = fig.add_subplot(gs[1, idx])
        ax_theta.plot(timestamps, theta_errors, '-', color=dark_grey_color, linewidth=1)        
        # Set labels and ticks
        if idx == 0:
            ax_theta.set_ylabel(r'$\theta$ error [rad]')

        ax_curv = fig.add_subplot(gs[2, idx])
        ax_curv.plot(timestamps, k1_errors, '-', color=blue_color, label='seg1', linewidth=1)
        ax_curv.plot(timestamps, k2_errors, '-', color=red_color, label='seg2', linewidth=1)         
        # Set labels and ticks
        if idx == 0:
            ax_curv.set_ylabel(r'Curvature error [$m^{-1}$]')
            ax_curv.legend()

        ax_temp = fig.add_subplot(gs[3, idx])
        ax_temp.plot(temp_timestamps, k1_temp, '-', color=blue_color, label='seg1', linewidth=1)
        ax_temp.plot(temp_timestamps, k2_temp, '-', color=red_color, label='seg2', linewidth=1)         
        # Set labels and ticks
        if idx == 0:
            ax_temp.set_ylabel('Temperature [C]')
            ax_temp.legend()

        ax_stiff = fig.add_subplot(gs[4, idx])
        ax_stiff.plot(timestamps, k1_stiff, '-', color=blue_color, label='seg1', linewidth=1)
        ax_stiff.plot(timestamps, k2_stiff, color=red_color, label='seg2', linewidth=2, linestyle=(0, (5, 10)))         
        # Set labels and ticks
        ax_stiff.set_xlabel('Time [s]')
        if idx == 0:
            ax_stiff.set_ylabel('Stiffness')
            ax_stiff.legend()


    print(f'k1 heating avrg: {k1_heating_avrg/4}')
    print(f'k1 cooling avrg: {k1_cooling_avrg/4}')
    print(f'k2 heating avrg: {k2_heating_avrg/4}')
    print(f'k1 cooling avrg: {k2_cooling_avrg/4}')

    plt.savefig('rm_grasp_analysis.pdf', format='pdf', dpi=150, bbox_inches='tight')
    plt.show()

def process_transport_data(all_transport_data):
    fig = plt.figure(figsize=(14, 8))

    gs = fig.add_gridspec(4, 4, 
                         width_ratios=[1, 1, 1, 1],
                         wspace=0.18,
                         hspace=0.32)
    
    for idx in range(4):        
        # Get data for this experiment
        data = all_transport_data[idx]

        path = []
        for path_point in data['path']:
            path.append([path_point['x'], path_point['y'], path_point['yaw']])
        path = np.array(path)

        object_traj = []
        timestamps = []

        for data_item in data['tracking']:
            timestamps.append(data_item['time'])
            object_traj.append(data_item['object']['pose'])
        object_traj = np.array(object_traj)

        # Create parameter for path interpolation (normalized distance along the path)
        path_param = np.linspace(0, 1, len(path))
        
        # Interpolate path to match object trajectory timestamps
        path_interp_x = np.interp(np.linspace(0, 1, len(timestamps)), path_param, path[:, 0])
        path_interp_y = np.interp(np.linspace(0, 1, len(timestamps)), path_param, path[:, 1])
        path_interp_yaw = np.interp(np.linspace(0, 1, len(timestamps)), path_param, path[:, 2])

        # Calculate errors
        x_error = object_traj[:, 0] - path_interp_x
        y_error = object_traj[:, 1] - path_interp_y
        
        # For yaw error, we need to handle angular differences correctly
        yaw_error = np.zeros_like(path_interp_yaw)
        for i in range(len(yaw_error)):
            # Calculate smallest angular difference
            diff = object_traj[i, 2] - path_interp_yaw[i]
            yaw_error[i] = (diff + np.pi) % (2 * np.pi) - np.pi

        ax_path = fig.add_subplot(gs[0, idx])
        ax_path.plot(100*path[:, 1], 100*path[:, 0], color=dark_grey_color, label='Path', linewidth=1)
        ax_path.plot(100*object_traj[:, 1], 100*object_traj[:, 0], color=red_color, label='Meas', linewidth=1)
        
        ax_path.set_xlabel('y [cm]')
        if idx == 0:
            ax_path.set_ylabel('x [cm]')
            ax_path.legend()
        
        # Plot errors
        ax_x = fig.add_subplot(gs[1, idx])
        
        # Plot position errors
        ax_x.plot(timestamps, x_error * 100, '-', color=dark_grey_color, linewidth=1)
        
        if idx == 0:
            ax_x.set_ylabel(r'$x$ error [cm]')

        ax_y = fig.add_subplot(gs[2, idx])
        ax_y.plot(timestamps, y_error * 100, '-', color=dark_grey_color, linewidth=1)
        
        if idx == 0:
            ax_y.set_ylabel(r'$y$ error [cm]')

        ax_th = fig.add_subplot(gs[3, idx])
        ax_th.plot(timestamps, yaw_error, '-', color=dark_grey_color, linewidth=1)
        
        ax_th.set_xlabel('Time [s]')
        if idx == 0:
            ax_th.set_ylabel(r'$\theta$ error [rad]')


    plt.savefig('rm_transport_analysis.pdf', format='pdf', dpi=150, bbox_inches='tight')
    plt.show()

        

all_traverse_data, all_transport_data = read_rm_files()

# process_traverse_data(all_traverse_data)
process_grasp(all_traverse_data)
# process_transport_data(all_transport_data)
