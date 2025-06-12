import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

names = ['circle (c)', 'ellipse (e)', 'heart (h)', 'bean (b)', 'heart g (h-g)']
names_boxplot = ['c', 'e', 'h', 'b', 'h-g']
trial_idx = [2, 1, 2, 0]
img_idx = [2, 2, 3, 2]


# sns.set_style("whitegrid", {'axes.grid' : False})

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['text.usetex'] = False
plt.rcParams['font.size'] = 26
# plt.rcParams['font.size'] = 10

colors = ['#dd8f07', 'k', '#d06261', '#0173b2']
line_width = 2.5  # Common variable for line width


def transport_files():
    directory = 'Experiments/Data/Tracking/Transport'

    cheescake_files = [f for f in os.listdir(directory) if f.startswith('cheescake') and f.endswith('.json')]
    ellipse_files = [f for f in os.listdir(directory) if f.startswith('ellipse') and f.endswith('.json')]
    heart_files = [f for f in os.listdir(directory) if f.startswith('heart') and f.endswith('.json')]
    bean_files = [f for f in os.listdir(directory) if f.startswith('bean') and f.endswith('.json')]
    
    cheescake_data = []
    ellipse_data = []
    heart_data = []
    bean_data = []

    for file in cheescake_files:
        file_path = os.path.join(directory, file)
        with open(file_path, 'r') as f:
            file_data = json.load(f)
            cheescake_data.append([file_data['path'], file_data['tracking']])

    for file in ellipse_files:
        file_path = os.path.join(directory, file)
        with open(file_path, 'r') as f:
            file_data = json.load(f)
            ellipse_data.append([file_data['path'], file_data['tracking']])

    for file in heart_files:
        file_path = os.path.join(directory, file)
        with open(file_path, 'r') as f:
            file_data = json.load(f)
            heart_data.append([file_data['path'], file_data['tracking']])

    for file in bean_files:
        file_path = os.path.join(directory, file)
        with open(file_path, 'r') as f:
            file_data = json.load(f)
            bean_data.append([file_data['path'], file_data['tracking']])

    return cheescake_data, ellipse_data, heart_data, bean_data

def movig_avrg(x, y, window_size=10):
    y = np.array(y)
    smoothed_y = np.convolve(y, np.ones(window_size)/window_size, mode='valid')
    # Adjust timestamps to match the smoothed data length
    smoothed_x = x[window_size-1:]

    return smoothed_x, smoothed_y

def traverse_files():
    directory = 'Experiments/Data/Tracking/Traverse'
    all_files = [f for f in os.listdir(directory) if f.endswith('.json')]

    transport_data = []

    for file in all_files:
        file_path = os.path.join(directory, file)
        with open(file_path, 'r') as f:
            file_data = json.load(f)
            transport_data.append([file_data['transport']['path'], 
                                   file_data['transport']['tracking'], 
                                   file_data['traversing']])

    return transport_data

def calc_tracking_error(entry):
    path_data = entry[0]
    tracking_data = entry[1]

    # Process reference path
    path = []
    for path_point in path_data:
        path.append([path_point['x'], path_point['y'], path_point['yaw']])
    path = np.array(path)[:, :2]  # Only take x,y coordinates

    # Process tracking data
    timestamps = []
    object_traj = []
    robot_traj = []

    for item in tracking_data:
        timestamps.append(item['time'])
        object_traj.append(item['object']['pose'])
        robot_traj.append(item['robot']['pose'])
    
    # Convert to numpy arrays
    timestamps = np.array(timestamps)
    object_traj = np.array(object_traj)  # Only take x,y coordinates
    
    # Calculate tracking error at each timestep
    errors = []
    for obj_pos in object_traj[:,:2]:
        # Calculate distances to all points in the path
        distances = np.linalg.norm(path - obj_pos, axis=1)
        # Get minimum distance
        min_distance = np.min(distances)
        errors.append(1000 * min_distance)
    
    return movig_avrg(timestamps, errors)

def plot_pos(idx):
    ellipse_entry = data[idx][trial_idx[idx]]
    tracking_data = ellipse_entry[1]

    # Process tracking data
    timestamps = []
    object_traj = []
    robot_traj = []

    for item in tracking_data:
        timestamps.append(item['time'])
        object_traj.append(item['object']['pose'])
        robot_traj.append(item['robot']['pose'])
    
    # Convert to numpy arrays
    timestamps = np.array(timestamps)
    object_traj = np.array(object_traj)

    return timestamps, object_traj[:,0], object_traj[:,1]

def process_transport_data():    
    
    # Create figure with 2 vertical subplots
    fig = plt.figure(figsize=(16, 20))

    outer_gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.23)
    left_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_gs[0], height_ratios=[1, 4], hspace=0.2)
    block2_l_gs = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=left_gs[1], hspace=0.1)
    right_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_gs[1], height_ratios=[3, 2], hspace=0.13)
    block2_r_gs = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=right_gs[0], hspace=0.25)

    ax_th = fig.add_subplot(left_gs[0])

    # First subplot
    for shape_data, name, trial, img_id, color in zip(data, names, trial_idx, img_idx, colors):
        entry = shape_data[trial]
        tracking_data = entry[1]

        # Process tracking data
        timestamps = []
        object_traj = []
        robot_traj = []

        for item in tracking_data:
            timestamps.append(item['time'])
            object_traj.append(item['object']['pose'])
            robot_traj.append(item['robot']['pose'])
        
        # Convert to numpy arrays
        timestamps = np.array(timestamps)
        object_traj = np.array(object_traj)

        # Apply moving average to smooth the error data
        window_size = 5  # Adjust this value to control smoothing amount
        smoothed_theta = np.convolve(object_traj[:,2], np.ones(window_size)/window_size, mode='valid')
        # Adjust timestamps to match the smoothed data length
        smoothed_timestamps = timestamps[window_size-1:] - timestamps[0]

        # Normalize timestamps to range [0, 1]
        normalized_timestamps = (smoothed_timestamps - smoothed_timestamps[0]) / (smoothed_timestamps[-1] - smoothed_timestamps[0])
        object_traj = np.array(object_traj)
    
        ax_th.plot(normalized_timestamps, smoothed_theta, label=f'{name} {img_id}', color=color, linewidth=line_width)

    ax_th.set_ylabel(r'$\theta$ [rad]', labelpad=48)
    ax_th.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.5))
    ax_th.set_xlabel('Normalized Time')

    circle_idx = 0
    circle_entry = data[1][circle_idx]

    smoothed_timestamps, smoothed_errors = calc_tracking_error(circle_entry)

    ax_error = fig.add_subplot(block2_l_gs[0])
    ax_error.plot(smoothed_timestamps, smoothed_errors, color=colors[1], linewidth=line_width)

    ax_error.set_title(f'Ellipse 3')  # Set title for the subplot
    ax_error.set_ylabel('e [mm]', labelpad=48)  # Change this to match your data
    ax_error.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)  # Remove ticks in x axis


    tracking_data = circle_entry[1]

    # Process tracking data
    timestamps = []
    object_vel = []
    robot_vel = []

    for item in tracking_data:
        timestamps.append(item['time'])
        object_vel.append(item['object']['target_velocity'])
        robot_vel.append(item['robot']['target_velocity'])
    
    # Convert to numpy arrays
    timestamps = np.array(timestamps)
    object_vel = np.array(object_vel)
    robot_vel = np.array(robot_vel)

    ax_vel_y = fig.add_subplot(block2_l_gs[1])

    ax_vel_y.plot(timestamps, 100 * robot_vel[:,1], label='robot', color=colors[1], linewidth=line_width)
    ax_vel_y.plot(timestamps, 100 * object_vel[:,0], label='object', color=colors[3], linewidth=line_width)

    ax_vel_y.set_ylabel(r'$v_h$ [cm/s]', labelpad=48) 
    ax_vel_y.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)  # Remove ticks in x axis
    ax_vel_y.legend()

    ax_vel_x = fig.add_subplot(block2_l_gs[2])

    ax_vel_x.plot(timestamps, 100 * robot_vel[:,0], color=colors[1], linewidth=line_width)
    ax_vel_x.plot(timestamps, 100 * object_vel[:,1], color=colors[3], linewidth=line_width)

    # Second subplot (you can add your data here later)
    # ax2.set_xlabel('Time [s]')
    ax_vel_x.set_ylabel(r'$v_l$ [cm/s]', labelpad=26)  
    ax_vel_x.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)  # Remove ticks in x axis

    ax_vel_omega = fig.add_subplot(block2_l_gs[3])

    ax_vel_omega.plot(timestamps, 100 * robot_vel[:,2], color=colors[1], linewidth=line_width)
    ax_vel_omega.plot(timestamps, 100 * object_vel[:,2], color=colors[3], linewidth=line_width)

    # Second subplot (you can add your data here later)
    # ax2.set_xlabel('Time [s]')
    ax_vel_omega.set_xlabel('Time [s]')
    ax_vel_omega.set_ylabel(r'$\omega$ [rad/s] $\times 10^{-2}$', labelpad=9)  


    heart_theta_compare = bean_data[1]
    tracking_data = heart_theta_compare[1]

    timestamps = []
    object_traj = []
    robot_traj = []

    for item in tracking_data:
        timestamps.append(item['time'])
        object_traj.append(item['object']['pose'])
        robot_traj.append(item['robot']['pose'])
    
    # Convert to numpy arrays
    timestamps = np.array(timestamps)
    object_traj = np.array(object_traj)
    robot_traj = np.array(robot_traj)

    ax_heart_th = fig.add_subplot(block2_r_gs[0])

    ax_heart_th.plot(timestamps, robot_traj[:,2], label='robot', color=colors[1], lw=line_width)
    ax_heart_th.plot(timestamps, object_traj[:,2], label='object', color=colors[3], lw=line_width)

    # ax_heart_good.set_xlabel('Time [s]')
    ax_heart_th.set_ylabel(r'$\theta$ [rad]') 
    ax_heart_th.legend()


    heart_grasp_data = data[-1]
    heart_good_entry = heart_grasp_data[3]

    tracking_data = heart_good_entry[1]

    # Process tracking data
    timestamps = []
    object_traj = []
    robot_traj = []

    for item in tracking_data:
        timestamps.append(item['time'])
        object_traj.append(item['object']['pose'])
        robot_traj.append(item['robot']['pose'])
    
    # Convert to numpy arrays
    timestamps = np.array(timestamps)
    object_traj = np.array(object_traj)

    ax_heart_good = fig.add_subplot(block2_r_gs[1])
    ax_heart_good.plot(timestamps, 100 * object_traj[:,0], label='x', color=colors[3], lw=line_width)
    ax_heart_good.plot(timestamps, 100 * object_traj[:,1], label='y', color=colors[1], lw=line_width)

    ax_heart_good.set_ylabel('Position [cm]') 
    ax_heart_good.legend(ncol=2)


    smoothed_timestamps, smoothed_errors = calc_tracking_error(heart_good_entry)

    sub_good_ax = ax_heart_good.inset_axes([0.58, 0.22, 0.37, 0.37])
    sub_good_ax.plot(smoothed_timestamps, smoothed_errors, color=colors[2], linewidth=1)

    sub_good_ax.tick_params(axis='both', labelsize=14)  # Change ticks font size to 12
    sub_good_ax.yaxis.label.set_size(14)  # Set the font size of the y-axis label to 10
    sub_good_ax.set_ylabel('e [mm]') 


    heart_stabilized_entry = heart_grasp_data[5]
    tracking_data = heart_stabilized_entry[1]

    # Process tracking data
    timestamps = []
    object_traj = []
    robot_traj = []

    for item in tracking_data:
        timestamps.append(item['time'])
        object_traj.append(item['object']['pose'])
        robot_traj.append(item['robot']['pose'])
    
    # Convert to numpy arrays
    timestamps = np.array(timestamps)
    object_traj = np.array(object_traj)

    ax_heart_stabilized = fig.add_subplot(block2_r_gs[2])
    ax_heart_stabilized.plot(timestamps, 100 * object_traj[:,0], label='x', color=colors[3], lw=line_width)
    ax_heart_stabilized.plot(timestamps, 100 * object_traj[:,1], label='y', color=colors[1], lw=line_width)

    ax_heart_stabilized.set_xlabel('Time [s]')
    ax_heart_stabilized.set_ylabel('Position [cm]') 
    ax_heart_stabilized.legend(ncol=2)

    smoothed_timestamps, smoothed_errors = calc_tracking_error(heart_stabilized_entry)

    sub_stab_ax = ax_heart_stabilized.inset_axes([0.6, 0.22, 0.37, 0.37])
    sub_stab_ax.plot(smoothed_timestamps, smoothed_errors, color=colors[2], linewidth=1)

    sub_stab_ax.tick_params(axis='both', labelsize=14)  # Change ticks font size to 12
    sub_stab_ax.yaxis.label.set_size(14)  # Set the font size of the y-axis label to 10
    sub_stab_ax.set_ylabel('e [mm]') 



    # Initialize empty DataFrame before the loop
    df_errors = pd.DataFrame(columns=['Tracking error e [mm]', 'Shape'])
    
    for shape_data, name in zip(data, names_boxplot):
        for data_entry in shape_data:
            smoothed_timestamps, smoothed_errors = calc_tracking_error(data_entry)
            # Create temporary DataFrame for this iteration
            temp_df = pd.DataFrame({
                'Tracking error e [mm]': [np.mean(smoothed_errors)],
                'Shape': [name]
            })
            # Concatenate with the main DataFrame
            df_errors = pd.concat([df_errors, temp_df], ignore_index=True)

    ax_boxplot = fig.add_subplot(right_gs[1])
    sns.boxplot(x='Shape', y='Tracking error e [mm]', data=df_errors, ax=ax_boxplot, color='#a2c9f4',
                linecolor='k')

    
    # Adjust spacing between subplots
    plt.tight_layout()
    plt.show()
    # plt.savefigs('transport.pdf', format='pdf', dpi=150, bbox_inches='tight', transparent=True)


def process_grasp_data():

    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 2, wspace=0.3, hspace=0.1)

    data_entry = heart_traverse_files[0]
    grasp_data = list(data_entry[2].values())[-1]

    timestamps = grasp_data['tracking']['time']
    target_config = grasp_data['target_config']
    robot_states = grasp_data['tracking']['robot_states']
    input_stiffness = grasp_data['tracking']['target_mm']
    input_vel = grasp_data['tracking']['target_vel']
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

    robot_states = np.array(robot_states)
    
    ax_pos = fig.add_subplot(gs[0, 0])
    ax_pos.plot(timestamps, 100 * robot_states[:,0], label='x', color='k', lw=line_width)
    ax_pos.plot(timestamps, 100 * robot_states[:,1], label='y', color=colors[3], lw=line_width)      

    ax_pos.set_ylabel('Position [cm]')
    ax_pos.legend()
    ax_pos.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)


    ax_th = fig.add_subplot(gs[1, 0])
    ax_th.plot(timestamps, 10 * robot_states[:,2], color='k', lw=line_width)

    ax_th.set_ylabel(r'$\theta$ [rad] $\times 10^{-1}$', labelpad=20)
    ax_th.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    ax_k = fig.add_subplot(gs[2, 0])
    ax_k.plot(timestamps, robot_states[:,3], label=r'$\kappa_1$', color='k', lw=line_width)
    ax_k.plot(timestamps, robot_states[:,4], label=r'$\kappa_2$', color=colors[3], lw=line_width)      

    ax_k.set_ylabel(r'Curv [$m^{-1}$]', labelpad=25)
    ax_k.legend()
    ax_k.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)


    ax_temp = fig.add_subplot(gs[3, 0])
    ax_temp.plot(temp_timestamps, k1_temp, label='seg 1', color=colors[1], lw=line_width)
    ax_temp.plot(temp_timestamps, k2_temp, label='seg 2', color=colors[2], lw=line_width) 

    ax_temp.set_xlabel('Time [s]')
    ax_temp.set_ylabel(r'Temp [$^{\circ}$C]', labelpad=25)
    ax_temp.legend(ncol=2)     

    input_vel = np.array(input_vel)

    ax_lin_vel = fig.add_subplot(gs[0, 1])
    ax_lin_vel.plot(timestamps, 100 * input_vel[:,0], label=r'v_x', color='k', lw=line_width)
    ax_lin_vel.plot(timestamps, 100 * input_vel[:,1], label=r'v_y', color=colors[3], lw=line_width)   

    ax_lin_vel.set_ylabel('Vel [cm/s]', labelpad=43)
    ax_lin_vel.legend()   
    ax_lin_vel.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    ax_rot_vel = fig.add_subplot(gs[1, 1])
    ax_rot_vel.plot(timestamps, 100 * input_vel[:,2], color='k', lw=line_width) 

    ax_rot_vel.set_ylabel(r'$\omega$ [rad/s] $\times 10^{-2}$')
    ax_rot_vel.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)


    ax_k_vel = fig.add_subplot(gs[2, 1])
    ax_k_vel.plot(timestamps, 100 * input_vel[:,3], label=r'v_1', color='k', lw=line_width)
    ax_k_vel.plot(timestamps, 100 * input_vel[:,4], label=r'v_2', color=colors[3], lw=line_width)   

    ax_k_vel.set_ylabel(r'Vel [$cm^{-1}$/s]', labelpad=20)
    ax_k_vel.legend(loc='upper right')  
    ax_k_vel.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    input_stiffness = np.array(input_stiffness)

    ax_stiff = fig.add_subplot(gs[3, 1])
    ax_stiff.plot(timestamps, input_stiffness[:,0], label='seg 1', color=colors[1], lw=line_width)
    ax_stiff.plot(timestamps, input_stiffness[:,1], label='seg 2', color=colors[2], lw=line_width, linestyle=(0, (5, 10))) 

    ax_stiff.set_xlabel('Time [s]')
    ax_stiff.set_ylabel('Input stiffness', labelpad=43)
    ax_stiff.set_yticks([0, 1])
    ax_stiff.legend() 

    plt.tight_layout()
    # plt.show()
    plt.savefig('grasp.pdf', format='pdf', dpi=150, bbox_inches='tight', transparent=True)

cheescake_data, ellipse_data, heart_data, bean_data = transport_files()
heart_traverse_files = traverse_files()
data = [cheescake_data, ellipse_data, heart_data, bean_data, heart_traverse_files]

process_transport_data()
# process_grasp_data()