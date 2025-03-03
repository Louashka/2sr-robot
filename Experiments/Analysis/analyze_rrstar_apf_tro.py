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

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['text.usetex'] = False
plt.rcParams['font.size'] = 24

colors = ['#dd8f07', 'k', '#d06261', '#0173b2']

obstacles_file_path = f'Experiments/Data/Tracking/Grasp/obstacles.json'

def read_traverse_rm_files():
    directory = 'Experiments/Data/Tracking/Grasp'
    traverse_rm_files = [f for f in os.listdir(directory) if f.startswith('traverse_rm_') and f.endswith('.json')]
    paths_data = []

    for file in traverse_rm_files:
        file_path = os.path.join(directory, file)
        with open(file_path, 'r') as f:
            file_data = json.load(f)
            paths_data.append(file_data['estimation']['robot_reference_path'])

    return paths_data

def read_images_from_directory():
    directory = 'Experiments/Figures/RRT_APF'
    image_files = [f for f in os.listdir(directory) if f.endswith(('.png', '.jpg', '.jpeg'))]
    images = []

    for image_file in image_files:
        image_path = os.path.join(directory, image_file)
        image = cv2.imread(image_path)
        if image is not None:
            images.append(image)

    return images

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

def transform_point(image_point, transform_matrix):
    """Transform a single point from image coordinates to global coordinates"""
    point = np.array([[[float(image_point[0]), float(image_point[1])]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(point, transform_matrix)
    return transformed[0][0].tolist()

def calibrate(circles, path):
    circles_points = []

    for circle in circles:
        circle_contour = circle[1]

        (x, y), _ = cv2.minEnclosingCircle(circle_contour)
        center = [int(x), int(y)]
        circles_points.append(center)

    # Convert to numpy arrays and reshape for perspective transform
    src_points = np.array(circles_points, dtype=np.float32)
    dst_points = np.array([[p[0], p[1]] for i, p in enumerate(path) if i % 2 == 0], 
                          dtype=np.float32)

    # Calculate the perspective transform matrix
    transform_matrix = cv2.findHomography(src_points, dst_points)[0]

    return transform_matrix

def process_data(paths, images):
    obstacles_data = {}
    idx = 1

    for path, image in zip(paths, images):
        # Convert the image to the HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define the range for red color in HSV
        # Adjusted thresholds for better red detection
        lower_red1 = np.array([0, 70, 50])     # More permissive saturation and value
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 70, 50])   # More permissive saturation and value
        upper_red2 = np.array([180, 255, 255])

        # Create masks for red color
        mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        # Find contours of the red circles
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter and sort circles based on size
        circles = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 0:  
                circles.append((area, contour))

        # Sort circles by area (size) in descending order
        circles.sort(key=lambda x: x[0], reverse=True)

        transform_matrix = calibrate(circles, path)
        print(transform_matrix)

        # plt.figure(figsize=(10, 6))

        # for circle in circles:
        #     circle_contour = circle[1]

        #     (x, y), _ = cv2.minEnclosingCircle(circle_contour)
        #     center = [int(x), int(y)]

        #     # point_transformed = transform_point(center, transform_matrix)
        #     # plt.plot([point_transformed[0]], [point_transformed[1]], 'o')

        #     for contour_points in circle_contour:
        #         point_transformed = transform_point(contour_points[0], transform_matrix)

        #         plt.plot([point_transformed[0]], [point_transformed[1]], 'r.')

        # --------------------------------------------------------------

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
                    obstacle.append(transform_point(corner[0], transform_matrix))

                obstacle_polygon = Polygon(obstacle)
                if obstacle_polygon.is_valid:
                    obstacles.append(obstacle_polygon)

                obstacles_corners.append(obstacle)

        print(len(obstacles))

        obstacles_data[idx] = obstacles_corners
        idx += 1

        # Create a figure for plotting
        plt.figure(figsize=(10, 6))

        path_points = np.array(path)
        plt.plot(path_points[:, 0], path_points[:, 1], marker='o', label='Path', color='blue')

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
        plt.show()

    data_json = {
        'rm': obstacles_data
    }

    with open(obstacles_file_path, 'w') as f:
        json.dump(data_json, f, indent=2)
    print(f"Data written to {obstacles_file_path}\n")

def get_obstacles():
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
    

def analyze_data(all_paths, all_obstacles):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15, 4.5))

    # Create an inset axes in the upper right corner of ax1
    # axins = ax1.inset_axes([0.6, 0.6, 0.35, 0.35])  # [x, y, width, height] in axes coordinates

    max_path_length = 0
    for idx, (path, obstacles) in enumerate(zip(all_paths, all_obstacles)):
        orientations = [pose[2] for pose in path]
        clearance_cm = []
        clearance_mm = []

        for pose in path[:-1]:
            robot_polygon = getRobotPolygon(pose)
            min_distance = float('inf')
            for obstacle in obstacles:
                distance = robot_polygon.distance(obstacle)
                min_distance = min(min_distance, distance)
            clearance_cm.append(min_distance * 100)  # Convert to cm
            clearance_mm.append(min_distance * 1000)

        print(clearance_mm)

        # Plot on main axes
        sns.lineplot(data=clearance_cm, ax=ax1, label=f'{idx+4}', color=colors[idx])
        sns.scatterplot(x=range(len(clearance_cm)), y=clearance_cm, ax=ax1, color=colors[idx], s=70)
        sns.lineplot(data=orientations, ax=ax2, color=colors[idx])
        sns.scatterplot(x=range(len(orientations)), y=orientations, ax=ax2, color=colors[idx], s=70)
        
        # Plot on inset axes
        # sns.lineplot(data=clearance_mm, ax=axins, color=colors[idx])

    # axins.set_xlim(2.5, 3.5)
    # axins.set_ylim(0, 30)

    # # Remove labels from inset
    # axins.set_xlabel('')
    # axins.set_ylabel('[mm]')
    
    # Main axes labels
    ax1.set_xlabel('Waypoints')
    ax1.set_ylabel('Clearance [cm]')
    ax1.legend(ncol=2)

    ax2.set_xlabel('Waypoints')
    ax2.set_ylabel('Orientation [rad]')
    # ax2.legend(ncol=2, loc='lower right')
    
    
    plt.tight_layout()
    plt.savefig('rrtstar_apf_planning.pdf', format='pdf', dpi=150, bbox_inches='tight', transparent=True)
    # plt.show()

paths_data = read_traverse_rm_files()
# images = read_images_from_directory()
# process_data(paths_data, images)

obstacles_data = get_obstacles()
analyze_data(paths_data, obstacles_data)