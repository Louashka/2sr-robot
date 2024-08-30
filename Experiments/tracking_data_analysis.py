import os
import glob
import json
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

def analyze_path_tracking(target_path, robot_tracking_data):
    # Interpolate target path
    path_length = np.cumsum(np.sqrt(np.sum(np.diff(target_path[:,:2], axis=0)**2, axis=1)))
    path_length = np.insert(path_length, 0, 0)
    path_interp = interp1d(path_length, target_path.T, kind='cubic', axis=1, bounds_error=False, fill_value="extrapolate")

    # Initialize error metrics
    lateral_errors = []
    longitudinal_errors = []
    orientation_errors = []
    
    # Initialize new metrics
    timestamps = []
    positions = []
    orientations = []
    curvatures = []
    stiffnesses = []
    velocities = []
    angular_velocities = []
    pose_errors = []
    velocity_errors = []
    distances_to_target = []

    num_points = 1000
    t = np.linspace(0, path_length[-1], num_points)
    points_on_path = path_interp(t).T

    for data_point in robot_tracking_data:
        timestamps.append(data_point['time'])
        
        robot_pose = [data_point['pose']['x'], data_point['pose']['y'], data_point['pose']['theta']]
        positions.append([robot_pose[0], robot_pose[1]])
        orientations.append(robot_pose[2])
        curvatures.append([data_point['pose']['k1'], data_point['pose']['k2']])
        stiffnesses.append([data_point['pose']['stiff1'], data_point['pose']['stiff2']])
        
        velocities.append([data_point['vel']['v_x'], data_point['vel']['v_y']])
        angular_velocities.append(data_point['vel']['omega'])
        
        pose_errors.append([data_point['errors']['pose_errors']['e_x'], 
                            data_point['errors']['pose_errors']['e_y'], 
                            data_point['errors']['pose_errors']['e_theta']])
        velocity_errors.append([data_point['errors']['vel_errors']['e_v_x'], 
                                data_point['errors']['vel_errors']['e_v_y'], 
                                data_point['errors']['vel_errors']['e_omega']])
        
        distances_to_target.append(data_point['distance_to_target'])
        
        # Find nearest point on path
        distances = cdist([robot_pose[:2]], points_on_path[:,:2])[0]
        nearest_index = np.argmin(distances)
        nearest_point = points_on_path[nearest_index]
        
        # Lateral error
        lateral_error = distances[nearest_index]
        lateral_errors.append(lateral_error)
        
        # Longitudinal error
        if nearest_index == num_points - 1:
            longitudinal_error = path_length[-1] - t[-1]
        else:
            longitudinal_error = path_length[-1] * nearest_index / (num_points - 1) - t[nearest_index]
        longitudinal_errors.append(longitudinal_error)
        
        # Orientation error
        target_yaw = nearest_point[2]
        orientation_error = np.abs(robot_pose[2] - target_yaw)
        orientation_error = min(orientation_error, 2*np.pi - orientation_error)
        orientation_errors.append(orientation_error)

    # Convert lists to numpy arrays for easier processing
    timestamps = np.array(timestamps)
    positions = np.array(positions)
    orientations = np.array(orientations)
    curvatures = np.array(curvatures)
    stiffnesses = np.array(stiffnesses)
    velocities = np.array(velocities)
    angular_velocities = np.array(angular_velocities)
    pose_errors = np.array(pose_errors)
    velocity_errors = np.array(velocity_errors)
    distances_to_target = np.array(distances_to_target)

    # Total path length error
    robot_path_length = np.sum(np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1)))
    path_length_error = np.abs(robot_path_length - path_length[-1])
    
    # Compute accelerations
    accelerations = np.diff(velocities, axis=0) / np.diff(timestamps)[:, np.newaxis]
    angular_accelerations = np.diff(angular_velocities) / np.diff(timestamps)
    
    return {
        'timestamps': timestamps,
        'positions': positions,
        'orientations': orientations,
        'curvatures': curvatures,
        'stiffnesses': stiffnesses,
        'velocities': velocities,
        'angular_velocities': angular_velocities,
        'accelerations': accelerations,
        'angular_accelerations': angular_accelerations,
        'pose_errors': pose_errors,
        'velocity_errors': velocity_errors,
        'distances_to_target': distances_to_target,
        'lateral_errors': lateral_errors,
        'longitudinal_errors': longitudinal_errors,
        'orientation_errors': orientation_errors,
        'path_length_error': path_length_error,
    }

def plot_path_tracking_results(target_path, results, json_file_name):
    # Set up the figure with subplots
    fig = plt.figure(figsize=(19, 12))
    fig.suptitle(json_file_name)
    
    # 1. Path Tracking
    ax1 = fig.add_subplot(221)
    ax1.plot(target_path[:, 0], target_path[:, 1], 'b-', label='Target Path')
    ax1.plot(results['positions'][:, 0], results['positions'][:, 1], 'r--', label='Robot Path')
    ax1.set_title('Path Tracking')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.legend()
    ax1.grid(True)

    # 2. Errors over time
    ax2 = fig.add_subplot(222)
    ax2.plot(results['timestamps'], results['lateral_errors'], label='Lateral Error')
    ax2.plot(results['timestamps'], results['longitudinal_errors'], label='Longitudinal Error')
    ax2.plot(results['timestamps'], results['orientation_errors'], label='Orientation Error')
    ax2.set_title('Tracking Errors over Time')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Error')
    ax2.legend()
    ax2.grid(True)

    # # 3. Velocities
    # ax3 = fig.add_subplot(323)
    # ax3.plot(results['timestamps'], results['velocities'][:, 0], label='X Velocity')
    # ax3.plot(results['timestamps'], results['velocities'][:, 1], label='Y Velocity')
    # ax3.plot(results['timestamps'], results['angular_velocities'], label='Angular Velocity')
    # ax3.set_title('Velocities over Time')
    # ax3.set_xlabel('Time (s)')
    # ax3.set_ylabel('Velocity')
    # ax3.legend()
    # ax3.grid(True)

    # # 4. Accelerations
    # ax4 = fig.add_subplot(324)
    # ax4.plot(results['timestamps'][1:], results['accelerations'][:, 0], label='X Acceleration')
    # ax4.plot(results['timestamps'][1:], results['accelerations'][:, 1], label='Y Acceleration')
    # ax4.plot(results['timestamps'][1:], results['angular_accelerations'], label='Angular Acceleration')
    # ax4.set_title('Accelerations over Time')
    # ax4.set_xlabel('Time (s)')
    # ax4.set_ylabel('Acceleration')
    # ax4.legend()
    # ax4.grid(True)

    # 5. Pose Errors
    ax5 = fig.add_subplot(223)
    ax5.plot(results['timestamps'], results['pose_errors'][:, 0], label='X Error')
    ax5.plot(results['timestamps'], results['pose_errors'][:, 1], label='Y Error')
    ax5.plot(results['timestamps'], results['pose_errors'][:, 2], label='Theta Error')
    ax5.set_title('Pose Errors over Time')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Error')
    ax5.legend()
    ax5.grid(True)

    # 6. Velocity Errors
    ax6 = fig.add_subplot(224)
    ax6.plot(results['timestamps'], results['velocity_errors'][:, 0], label='X Velocity Error')
    ax6.plot(results['timestamps'], results['velocity_errors'][:, 1], label='Y Velocity Error')
    ax6.plot(results['timestamps'], results['velocity_errors'][:, 2], label='Angular Velocity Error')
    ax6.set_title('Velocity Errors over Time')
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Error')
    ax6.legend()
    ax6.grid(True)

    # # 7. Curvatures
    # ax7 = fig.add_subplot(427)
    # ax7.plot(results['timestamps'], results['curvatures'][:, 0], label='Curvature 1')
    # ax7.plot(results['timestamps'], results['curvatures'][:, 1], label='Curvature 2')
    # ax7.set_title('Curvatures over Time')
    # ax7.set_xlabel('Time (s)')
    # ax7.set_ylabel('Curvature')
    # ax7.legend()
    # ax7.grid(True)

    # # 8. Stiffnesses
    # ax8 = fig.add_subplot(428)
    # ax8.plot(results['timestamps'], results['stiffnesses'][:, 0], label='Stiffness 1')
    # ax8.plot(results['timestamps'], results['stiffnesses'][:, 1], label='Stiffness 2')
    # ax8.set_title('Stiffnesses over Time')
    # ax8.set_xlabel('Time (s)')
    # ax8.set_ylabel('Stiffness')
    # ax8.legend()
    # ax8.grid(True)

    plt.tight_layout()
    plt.show()

    # # Additional plot: Distance to target over time
    # plt.figure(figsize=(10, 6))
    # plt.plot(results['timestamps'], results['distances_to_target'])
    # plt.title('Distance to Target over Time')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Distance')
    # plt.grid(True)
    # plt.show()

    # Print some summary statistics
    print(f"Total path length error: {results['path_length_error']:.2f}")
    print(f"Mean lateral error: {np.mean(results['lateral_errors']):.2f}")
    print(f"Mean longitudinal error: {np.mean(results['longitudinal_errors']):.2f}")
    print(f"Mean orientation error: {np.mean(results['orientation_errors']):.2f}")
    print(f"Mean Velocity: {np.mean(results['velocities'])}")
    print(f"Mean Acceleration: {np.mean(results['accelerations'])}")
    print()
    print()

current_dir = os.path.dirname(__file__)
json_dir = os.path.join(current_dir, 'Data/')

json_file_names = glob.glob(os.path.join(json_dir, '*tracking_data_2024*.json'))

for json_file_name in json_file_names:
    json_file = open(json_file_name)
    data = json.load(json_file)

    target_path = np.array([[p['x'], p['y'], p['yaw']] for p in data['path']])

    results = analyze_path_tracking(target_path, data['robot_tracking'])
    plot_path_tracking_results(target_path, results, json_file_name)




    
    
