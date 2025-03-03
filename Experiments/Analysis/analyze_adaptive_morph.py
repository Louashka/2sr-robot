import sys
sys.path.append('D:/Robot 2SR/2sr-swarm-control')
import os
import json
import cv2
import numpy as np
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from Model import global_var as gv

agent_width = 0.07
agent_length = 0.32

colors = ['#dd8f07', '#d06261', 'k', '#0173b2']

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['text.usetex'] = False
plt.rcParams['font.size'] = 14

marker_size = 50

obstacles_file_path = f'Experiments/Data/Tracking/Grasp/obstacles.json'
camera_file_path = 'Controller/calibration_data.json'

w = 1280
h = 720

def read_camera_calibration_data():
    global camera_matrix, dist, R, tvec

    try:
        with open(camera_file_path, "r") as json_file:
            data = json.load(json_file)
    except FileNotFoundError:
        print(f"Error: {camera_file_path} not found.")
        return

    camera_data = data["camera"]

    fx = camera_data["fx"]
    fy = camera_data["fy"]
    cx = camera_data["cx"]
    cy = camera_data["cy"]
    k1 = camera_data["k1"]
    k2 = camera_data["k2"]
    p1 = camera_data["p1"]
    p2 = camera_data["p2"]
    k3 = camera_data["k3"]
    camera_matrix = np.array([[fx, 0, cx],
                                [0, fy, cy],
                                [0, 0, 1]])
    dist = np.array([k1, k2, p1, p2, k3])
    R = np.array(data["R"])
    tvec = np.array(data["tvec"]).reshape(3,1)

def imageToCamera(image_point, depth):
    u, v = image_point
    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist, (w,h), 1, (w,h))

    fx = new_camera_matrix[0, 0]
    fy = new_camera_matrix[1, 1]
    cx = new_camera_matrix[0, 2]
    cy = new_camera_matrix[1, 2]

    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth

    return np.array([x, y, z])

def cameraToGlobal(point_camera):
    """
    Convert camera coordinates to global coordinates.
    
    :param point_camera: 3D point in camera coordinates
    :return: 3D point in global coordinates
    """
    point_camera = np.array(point_camera).reshape(3, 1)
    point_global = np.linalg.inv(R) @ (point_camera - tvec)
    return point_global.flatten()

def imageToGlobal(image_point, depth):
    """
    Convert image coordinates to global coordinates.
    
    :param image_point: (u, v) coordinates in the image
    :param depth: The depth (Z coordinate) of the point in camera space
    :return: 3D point in global coordinates
    """
    point_camera = imageToCamera(image_point, depth)
    point_global = cameraToGlobal(point_camera)
    return point_global.tolist()


def read_images_from_directory():
    directory = 'Experiments/Figures/MVPP'
    image_files = [f for f in os.listdir(directory) if f.endswith(('.png', '.jpg', '.jpeg'))]
    images = []

    for image_file in image_files:
        image_path = os.path.join(directory, image_file)
        image = cv2.imread(image_path)
        if image is not None:
            images.append(image)

    return images

def extractObstacles():
    read_camera_calibration_data()
    images = read_images_from_directory()

    obstacles_data = {}
    idx = 1

    for image in images:
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define the range for blue color in HSV
        lower_blue = np.array([100, 150, 0])
        upper_blue = np.array([140, 255, 255])

        # Create a mask for blue color
        blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

        # Find contours of the blue polygons
        blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        obstacles = []
        obstacles_corners = []
    
        for contour in blue_contours:
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) > 3:
                obstacle = []

                for corner in approx:
                    obstacle.append(imageToGlobal(corner[0], 1.0))

                obstacle_polygon = Polygon(obstacle)
                if obstacle_polygon.is_valid:
                    obstacles.append(obstacle_polygon)

                obstacles_corners.append(obstacle)

        print(len(obstacles))

        obstacles_data[idx] = obstacles_corners
        idx += 1

        # Create a figure for plotting
        plt.figure(figsize=(10, 6))

        # Plot the obstacles
        for obstacle in obstacles:
            if obstacle.is_valid:
                x, y = obstacle.exterior.xy
                plt.fill(x, y, alpha=0.5, fc='red', ec='black', label='Obstacle')

        plt.title('Path Points and Obstacles')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.legend()
        plt.grid()
        plt.axis('equal')
        # plt.show()


    with open(obstacles_file_path, 'r+') as f:
        file_data = json.load(f)
        file_data['sm'] = obstacles_data
        f.seek(0)
        json.dump(file_data, f, indent=2)
    print(f"Data written to {obstacles_file_path}\n")

def readTraverseFiles():
    directory = 'Experiments/Data/Tracking/Grasp'
    traverse_rm_files = [f for f in os.listdir(directory) if f.startswith('traverse_sm_') and f.endswith('.json')]
    
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

        rm_experiments = obstacles_file_data['sm']
        for exp in rm_experiments.values():
            obstacles_polygons = []
            
            for obstacle_corners in exp:
                obstacles_polygons.append(Polygon(obstacle_corners))
            
            all_experiments.append(obstacles_polygons)

    return all_experiments

def arc(config, seg=1):
        total_l = gv.L_VSS
        th0 = config[2]
        k = config[2+seg]
        l = np.linspace(0, total_l, 10)

        flag = -1 if seg == 1 else 1
        theta_array = th0 + flag * k * l

        if k == 0:
            x = np.array([0, flag * total_l * np.cos(th0)])
            y = np.array([0, flag * total_l * np.sin(th0)])
        else:
            x = np.sin(theta_array) / k - np.sin(th0) / k
            y = -np.cos(theta_array) / k + np.cos(th0) / k

        x += config[0]
        y += config[1]

        points = []
        for x_i, y_i in zip(x, y):
            points.append([x_i, y_i])

        return points

def getRobotPolygon(config):
    """Create robot polygon for curved segments"""
    def generate_offset_points(points, width):
        # Convert points to numpy array if not already
        points = np.array(points).reshape(-1, 2)
        
        # Calculate tangent vectors between consecutive points
        tangents = np.roll(points, -1, axis=0) - points
        tangents = tangents[:-1]  # Remove last one
        
        # Normalize tangents
        lengths = np.sqrt(np.sum(tangents**2, axis=1))
        tangents = tangents / lengths[:, np.newaxis]
        
        # Calculate normal vectors (rotate tangents by 90 degrees)
        normals = np.array([-tangents[:, 1], tangents[:, 0]]).T
        
        # Generate offset points
        offset = width / 2
        upper_points = points[:-1] + normals * offset
        lower_points = points[:-1] - normals * offset
        
        return upper_points, lower_points

    points1 = arc(config, seg=1)
    points2 = arc(config, seg=2)
    
    # Generate offset points for both segments
    upper1, lower1 = generate_offset_points(points1, agent_width)
    upper2, lower2 = generate_offset_points(points2, agent_width)
    
    # Combine points to form a complete polygon
    # Go around the robot outline: upper1 -> upper2 -> lower2 (reversed) -> lower1 (reversed)
    polygon_points = np.vstack([
        upper1,
        upper2,
        lower2[::-1],
        lower1[::-1]
    ])
    
    # Create and return the polygon
    robot = Polygon(polygon_points)
    return robot

def analyze_data():
    fig, axs = plt.subplots(2, 5, figsize=(18,4))

    all_clearance = []

    for idx in range(4):        
        # Get data for this experiment
        data = all_traverse_data[idx]

        target_x = []
        target_y = []
        target_th = []
        target_k1 = []
        target_k2 = []
        target_timestamps = []

        target_idx = []
        last_idx = 0

        robot_x = []
        robot_y = []
        robot_th = []
        robot_k1 = []
        robot_k2 = []
        robot_timestamps = []

        temp1 = []
        temp2 = []
        temp_timestamps = []

        vel_x = []
        vel_y = []
        vel_omega = []
        vel1 = []
        vel2 = []

        stiff1_input = []
        stiff2_input = []

        obstacles = all_obstacles[idx]

        for data_item in list(data.values())[:-1]:
            timestamps = data_item['tracking']['time']
            target_config = data_item['target_config']
            robot_states = data_item['tracking']['robot_states']
            robot_vel = data_item['tracking']['target_vel']
            input_stiffness = data_item['tracking']['target_mm']
            temp_data = data_item['tracking']['temperature']

            target_x.append(target_config[0])
            target_y.append(target_config[1])
            target_th.append(target_config[2])
            target_k1.append(target_config[3])
            target_k2.append(target_config[4])

            target_timestamps.append(timestamps[-1])

            last_idx += len(robot_states)
            target_idx.append(last_idx)

            robot_x.extend([state[0] for state in robot_states])
            robot_y.extend([state[1] for state in robot_states])
            robot_th.extend([state[2] for state in robot_states])
            robot_k1.extend([state[3] for state in robot_states])
            robot_k2.extend([state[4] for state in robot_states])
            robot_timestamps.extend(timestamps)

            vel_x.extend([100 * vel[0] for vel in robot_vel])
            vel_y.extend([100 * vel[1] for vel in robot_vel])
            vel_omega.extend([10 * vel[2] for vel in robot_vel])
            vel1.extend([100 * vel[3] for vel in robot_vel])
            vel2.extend([100 * vel[4] for vel in robot_vel])

            for i in range(len(temp_data)):
                if temp_data[i] is not None:
                    rel_time = np.array(temp_data[i]['relative_timestamps']) + timestamps[i]
                    temp_timestamps.extend(rel_time.tolist())

                    temp1.extend([temp[0] for temp in temp_data[i]['meas']])
                    temp2.extend([temp[1] for temp in temp_data[i]['meas']])

            input_stiffness = np.array(input_stiffness)

            stiff1_input.extend(input_stiffness[:,0].tolist())
            stiff2_input.extend(input_stiffness[:,1].tolist())


        clearance = []
        
        for x, y, th, k1, k2 in zip(robot_x, robot_y, robot_th, robot_k1, robot_k2):
            robot_polygon = getRobotPolygon((x, y, th, k1, k2))
            min_distance = float('inf')
            for obstacle in obstacles:
                distance = robot_polygon.distance(obstacle)
                min_distance = min(min_distance, distance)
            clearance.append(100 * min_distance + 0.13)

        all_clearance.append(clearance)

        print(temp_timestamps[-1])
        
        if idx == 3:         
            robot_x = 100 * np.array(robot_x)
            robot_y = 100 * np.array(robot_y)

            target_x = 100 * np.array(target_x)
            target_y = 100 * np.array(target_y)

            robot_k1 = np.array(robot_k1)
            robot_k2 = np.array(robot_k2)
            temp1 = np.array(temp1)
            temp2 = np.array(temp2)

            robot_k1_filtered = pd.Series(robot_k1).rolling(window=10, center=True).median()
            robot_k2_filtered = pd.Series(robot_k2).rolling(window=10, center=True).median()
            temp1_filtered = pd.Series(temp1).rolling(window=10, center=True).median()
            temp2_filtered = pd.Series(temp2).rolling(window=10, center=True).median()

            axs[0,0].plot(robot_x, color=colors[2], lw=1, label='x')
            axs[0,0].scatter(target_idx, target_x, color=colors[2], marker='*', s=marker_size, label=r'x$_d$')

            axs[0,0].plot(robot_y, color=colors[3], lw=1, label='y')
            axs[0,0].scatter(target_idx, target_y, color=colors[3], marker='*', s=marker_size, label=r'y$_d$')

            axs[0,1].plot(robot_th, color=colors[2], lw=1)
            axs[0,1].scatter(target_idx, target_th, color=colors[2], marker='*', s=marker_size)

            axs[0,2].plot(robot_k1_filtered, color=colors[2], lw=1, label='VSS1')
            axs[0,2].scatter(target_idx, target_k1, color=colors[2], marker='*', s=marker_size)

            axs[0,2].plot(robot_k2_filtered, color=colors[1], lw=1, label='VSS2')
            axs[0,2].scatter(target_idx, target_k2, color=colors[1], marker='*', s=marker_size)

            axs[0,3].plot(temp_timestamps, temp1_filtered, color=colors[2], lw=1)
            axs[0,3].plot(temp_timestamps, temp2_filtered, color=colors[1], lw=1)

            axs[0,4].plot(clearance, color=colors[2], lw=1)
            
            # Add inset axes for the zoomed plot
            axins = axs[0,4].inset_axes([0.5, 0.5, 0.45, 0.45])  # [x, y, width, height] in axes coordinates
            
            # Plot the same data in the inset
            axins.plot(clearance, color=colors[2], lw=1)
            
            # Set the limits for the zoomed region (adjust these values as needed)
            x1, x2 = 210, 310  # Example x-range to zoom
            y1, y2 = -0.2, 4.9      # Example y-range to zoom
            axins.set_xlim(x1, x2)
            axins.set_ylim(y1, y2)
            
            axins.tick_params(axis='both', which='major', labelsize=9)  # Reduce font size of axis ticks

            axs[1,0].plot(vel_x, color=colors[2], lw=1, label=r'$v_x$')
            axs[1,0].plot(vel_y, color=colors[3], lw=1, label=r'$v_y$')

            axs[1,1].plot(vel_omega, color=colors[2], lw=1)

            axs[1,2].plot(vel1, color=colors[2], lw=1)
            axs[1,2].plot(vel2, '--', color=colors[1], lw=1)

            axs[1,3].plot(robot_timestamps, stiff2_input, color=colors[1], lw=1)
            axs[1,3].plot(robot_timestamps, stiff1_input, '--', color=colors[2], lw=1)


    axs[0,0].set_ylabel('Pos [cm]')
    axs[0,1].set_ylabel(r'$\theta$ [rad]')
    axs[0,2].set_ylabel(r'$\kappa_j$ [m$^{-1}$]')
    axs[0,3].set_ylabel(r'$T_j$ [$^{\circ}$C]')
    axs[0,4].set_ylabel('Clearance [cm]')
    

    axs[1,0].set_xlabel('Index')
    axs[1,0].set_ylabel(r'Vel [cm$^{-1}$]')

    axs[1,1].set_xlabel('Index')
    axs[1,1].set_ylabel(r'$\omega$ [rad/s] $\times 10^{-1}$')

    axs[1,2].set_xlabel('Index')
    axs[1,2].set_ylabel(r'$v_j$ [cm$^{-1}$]')

    axs[1,3].set_xlabel('Time [s]')
    axs[1,3].set_ylabel('Input stiff')
    axs[1,3].set_yticks([0, 1])


    axs[0,0].legend(ncol=2, prop={'size': 9})
    axs[0,2].legend(ncol=2, prop={'size': 9})
    axs[1,0].legend(loc='upper left', ncol=2, prop={'size': 9})

    axs[1,0].set_ylim([-10, 15])
    axs[1,3].set_ylim([-0.5, 1.5])

    sns.set_theme(palette="pastel")

    clearance_df = pd.DataFrame({
        'exp trial': np.repeat(['8', '9', '10', '11'], [len(c) for c in all_clearance]),
        'Clearance [cm]': np.concatenate(all_clearance)
    })
    sns.boxplot(x='exp trial', y='Clearance [cm]', data=clearance_df, ax=axs[1,4], 
                color='b', fliersize=0)


    axs[1,4].set_ylim([-5, 48])

    plt.tight_layout()
    # plt.savefig('sm_trav_analysis.pdf', format='pdf', dpi=150, bbox_inches='tight', transparent=True)
    plt.show()

# extractObstacles()

all_obstacles = getObstacles()
all_traverse_data = readTraverseFiles()

analyze_data()