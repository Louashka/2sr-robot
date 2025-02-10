import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

blue_color = '#0E595C'
red_color = '#DC5956'
dark_grey_color = '#333333'

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['text.usetex'] = False
plt.rcParams['font.size'] = 10


def transport_files():
    directory = 'Experiments/Data/Tracking/Object_transport'

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

# Calculate Frechet distance
def euc_dist(pt1, pt2):
    return np.sqrt(np.sum((pt1 - pt2) ** 2))

def _c(ca, i, j, P, Q):
    if ca[i, j] > -1:
        return ca[i, j]
    elif i == 0 and j == 0:
        ca[i, j] = euc_dist(P[0], Q[0])
    elif i > 0 and j == 0:
        ca[i, j] = max(_c(ca, i-1, 0, P, Q), euc_dist(P[i], Q[0]))
    elif i == 0 and j > 0:
        ca[i, j] = max(_c(ca, 0, j-1, P, Q), euc_dist(P[0], Q[j]))
    elif i > 0 and j > 0:
        ca[i, j] = max(
            min(_c(ca, i-1, j, P, Q),
                _c(ca, i-1, j-1, P, Q),
                _c(ca, i, j-1, P, Q)),
            euc_dist(P[i], Q[j]))
    else:
        ca[i, j] = float("inf")
    return ca[i, j]

def frechet_distance(P, Q):
    ca = np.ones((len(P), len(Q)))
    ca = np.multiply(ca, -1)
    return _c(ca, len(P)-1, len(Q)-1, P, Q)

def process_data(data):
    fig = plt.figure(figsize=(14, 7.5))

    gs = fig.add_gridspec(4, 4, 
                         width_ratios=[1, 1, 1, 1],
                         wspace=0.20,
                         hspace=0.32)
    
    idx = [0, 2, 0, 2]
    titles = ['Circle', 'Ellipse', 'Heart', 'Bean']
    
    for j in range(4): 
        shape_data = data[j]

        path_data = shape_data[idx[j]][0]
        tracking_data = shape_data[idx[j]][1]

        path = []
        for path_point in path_data:
            path.append([path_point['x'], path_point['y'], path_point['yaw']])
        path = np.array(path)

        timestamps = []
        object_traj = []
        robot_traj = []

        for item in tracking_data:
            timestamps.append(item['time'])
            object_traj.append(item['object']['pose'])
            robot_traj.append(item['robot']['pose'])

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

        # ---------------------------------------------------

        obj_theta_array = object_traj[:,2]
        timestamps = np.array(timestamps)

        # Calculate time differences
        dt = np.diff(timestamps)
        
        # Calculate theta differences (handling circular nature of angles)
        dtheta = np.diff(obj_theta_array)
        # Adjust for circular discontinuity (e.g., jump between -pi and pi)
        dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
        
        # Calculate angular velocity in rad/s
        angular_velocity = dtheta / dt
        
        # Convert to deg/s if desired
        obj_angular_velocity_deg = np.rad2deg(angular_velocity)

        # -------------------------------------------------------

        robot_theta_array = np.array(robot_traj)[:,2]
        
        # Calculate theta differences (handling circular nature of angles)
        robot_dtheta = np.diff(robot_theta_array)
        # Adjust for circular discontinuity (e.g., jump between -pi and pi)
        robot_dtheta = (robot_dtheta + np.pi) % (2 * np.pi) - np.pi
        
        # Calculate angular velocity in rad/s
        robot_angular_velocity = robot_dtheta / dt
        
        # Convert to deg/s if desired
        robot_angular_velocity_deg = np.rad2deg(robot_angular_velocity)

        
        # Plot errors
        ax_x = fig.add_subplot(gs[0, j])
        
        # Plot position errors
        ax_x.plot(timestamps, x_error * 100, '-', color=dark_grey_color, linewidth=1)
        
        if j == 0:
            ax_x.set_ylabel(r'$x$ error [cm]')
        ax_x.set_title(titles[j])

        ax_y = fig.add_subplot(gs[1, j])
        ax_y.plot(timestamps, y_error * 100, '-', color=dark_grey_color, linewidth=1)
        
        if j == 0:
            ax_y.set_ylabel(r'$y$ error [cm]')

        ax_th = fig.add_subplot(gs[2, j])
        ax_th.plot(timestamps, yaw_error, '-', color=dark_grey_color, linewidth=1)
        
        if j == 0:
            ax_th.set_ylabel(r'$\theta$ error [rad]')

        ax_vel = fig.add_subplot(gs[3, j])
        # Plot at the midpoints of the timestamps
        plot_times = (timestamps[1:] + timestamps[:-1]) / 2
        ax_vel.plot(plot_times, obj_angular_velocity_deg, color=blue_color, label='object', linewidth=1)
        ax_vel.plot(plot_times, robot_angular_velocity_deg, '--', color=red_color, label='robot', linewidth=1)

        if j == 0:
            ax_vel.set_ylabel('Ang vel [deg/s]')
            ax_vel.legend()
        ax_vel.set_xlabel('Time [s]')

    # plt.savefig('transport_analysis.pdf', format='pdf', dpi=150, bbox_inches='tight')
    plt.show()

cheescake_data, ellipse_data, heart_data, bean_data = transport_files()
process_data([cheescake_data, ellipse_data, heart_data, bean_data])

