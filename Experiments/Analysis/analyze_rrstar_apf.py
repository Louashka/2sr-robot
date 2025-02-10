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
agent_length = 0.34

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['text.usetex'] = False
plt.rcParams['font.size'] = 14

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

    file_path = f'Experiments/Data/Tracking/Grasp/obstacles.json'

    with open(file_path, 'w') as f:
        json.dump(data_json, f, indent=2)
    print(f"Data written to {file_path}\n")

    

paths_data = read_traverse_rm_files()
images = read_images_from_directory()
# all_obstacles = process_data(paths_data, images)

# analyze_data(paths_data, all_obstacles)