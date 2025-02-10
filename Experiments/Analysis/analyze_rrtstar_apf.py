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

def calibrate_frames(path, circles):
    """Calibrate camera frame to global frame using path and detected circles"""
    if not circles or not path:
        return None, None
    
    # Get the first and last positions from path
    path_start = np.array([path[0][0], path[0][1]])
    path_end = np.array([path[-1][0], path[-1][1]])
    
    # Get centers of the two largest circles (assumed to be start and end points)
    if len(circles) < 2:
        return None, None
    
    circle_centers = []
    for _, contour in circles[:2]:  # Take two largest circles
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            circle_centers.append(np.array([cx, cy]))
    
    if len(circle_centers) < 2:
        return None, None
    
    # Calculate transformation parameters
    camera_vector = circle_centers[1] - circle_centers[0]
    global_vector = path_end - path_start
    
    # Calculate scale factor
    scale = np.linalg.norm(global_vector) / np.linalg.norm(camera_vector)
    
    # Calculate rotation angle
    camera_angle = np.arctan2(camera_vector[1], camera_vector[0])
    global_angle = np.arctan2(global_vector[1], global_vector[0])
    rotation_angle = global_angle - camera_angle
    
    # Translation offset
    translation = path_start - scale * np.array([
        circle_centers[0][0] * np.cos(rotation_angle) - circle_centers[0][1] * np.sin(rotation_angle),
        circle_centers[0][0] * np.sin(rotation_angle) + circle_centers[0][1] * np.cos(rotation_angle)
    ])
    
    return {
        'scale': scale,
        'rotation': rotation_angle,
        'translation': translation
    }

def transform_contour_to_global(contour, calibration):
    """Transform contour points from camera frame to global frame"""
    transformed_points = []
    for point in contour[:, 0, :]:
        # Scale
        scaled_point = point * calibration['scale']
        
        # Rotate
        rotated_point = np.array([
            scaled_point[0] * np.cos(calibration['rotation']) - scaled_point[1] * np.sin(calibration['rotation']),
            scaled_point[0] * np.sin(calibration['rotation']) + scaled_point[1] * np.cos(calibration['rotation'])
        ])
        
        # Translate
        global_point = rotated_point + calibration['translation']
        transformed_points.append(global_point)
    
    return transformed_points

def process_data(paths, images):

    all_obstacles = []

    for path, image in zip(paths, images):
        # Convert the image to the HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define the range for blue color in HSV
        lower_blue = np.array([100, 150, 0])
        upper_blue = np.array([140, 255, 255])

        # Create a mask for blue color
        blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

        # Find contours of the blue polygons
        blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #------------------------------------------------------------

        # Convert the image to the HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define the range for red color in HSV
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        # Create masks for red color
        mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
        red_mask = mask1 | mask2

        # Find contours of the red circles
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter and sort circles based on size
        circles = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 0:  # Only consider non-zero area
                circles.append((area, contour))

        # Sort circles by area (size) in descending order
        circles.sort(key=lambda x: x[0], reverse=True)

        #------------------------------------------------------------

        # Calibrate frames
        calibration = calibrate_frames(path, circles)
        if calibration is None:
            continue

        #------------------------------------------------------------
        
        obstacles = []
        for contour in blue_contours:
            if len(contour) >= 3:
                # Transform contour points to global frame
                transformed_points = transform_contour_to_global(contour, calibration)
                # Create polygon in global frame
                polygon = Polygon(transformed_points)
                if polygon.is_valid:
                    obstacles.append(polygon)

        all_obstacles.append(obstacles)

    return all_obstacles

def analyze_data(paths, all_obstacles):
    clearance_color = '#1E7175'
    orientation_color = '#DC5956'
    
    all_clearance = []
    all_orientations = []

    for path, obstacles in zip(paths, all_obstacles):
        clearance = []
        orientations = [pose[2] * 180/np.pi for pose in path]  # Convert to degrees

        for pose in path:
            robot_polygon = getRobotPolygon(pose)
            min_distance = float('inf')
            for obstacle in obstacles:
                distance = robot_polygon.distance(obstacle)
                min_distance = min(min_distance, distance)
            clearance.append(min_distance * 100)  # Convert to cm

        all_clearance.append(clearance)
        all_orientations.append(orientations)

    # Create the combined plot
    fig = plt.figure(figsize=(18, 5.5))
    gs1 = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1], wspace=0.05, right=0.64)
    gs2 = fig.add_gridspec(2, 1, left=0.68)

    # Create shared x-axes
    shared_ax_top = None
    shared_ax_bottom = None

    # Plot individual clearances
    for i in range(4):
        if i == 0:
            ax = fig.add_subplot(gs1[0, i])
            shared_ax_top = ax
            ax.set_ylabel('Clearance [cm]')
        else:
            ax = fig.add_subplot(gs1[0, i], sharey=shared_ax_top, sharex=shared_ax_top)
            ax.set_ylabel('')
            plt.setp(ax.get_yticklabels(), visible=False)

        sns.lineplot(data=all_clearance[i], ax=ax, color=clearance_color)
        sns.scatterplot(x=range(len(all_clearance[i])), y=all_clearance[i], ax=ax, color=clearance_color, s=20)
        ax.set_xlabel('')
        ax.tick_params(axis='y')

    # Plot individual orientations
    for i in range(4):
        if i == 0:
            ax = fig.add_subplot(gs1[1, i])
            shared_ax_bottom = ax
            ax.set_ylabel('Orientation [°]')
        else:
            ax = fig.add_subplot(gs1[1, i], sharey=shared_ax_bottom, sharex=shared_ax_bottom)
            ax.set_ylabel('')
            plt.setp(ax.get_yticklabels(), visible=False)

        sns.lineplot(data=all_orientations[i], ax=ax, color=orientation_color)
        sns.scatterplot(x=range(len(all_orientations[i])), y=all_orientations[i], ax=ax, color=orientation_color, s=20)
        ax.set_xlabel('Waypoints')
        ax.tick_params(axis='y')

    # Create data for boxplots
    clearance_data = pd.DataFrame({
        'Case': np.repeat(['Case 1', 'Case 2', 'Case 3', 'Case 4'], 
                         [len(c) for c in all_clearance]),
        'Clearance': np.concatenate(all_clearance)
    })

    orientation_data = pd.DataFrame({
        'Case': np.repeat(['Case 1', 'Case 2', 'Case 3', 'Case 4'], 
                         [len(o) for o in all_orientations]),
        'Orientation': np.concatenate(all_orientations)
    })

    # Plot boxplots
    ax_combined = fig.add_subplot(gs2[:, 0])
    
    # Calculate positions for offset boxplots
    positions = np.arange(4)
    width = 0.43
    
    # First y-axis for clearance
    sns.boxplot(data=clearance_data, x='Case', y='Clearance', ax=ax_combined, 
                color="#8AC2C5", positions=positions-width/2, gap=0.48)
    ax_combined.set_ylabel('Clearance [cm]', color=clearance_color)
    ax_combined.tick_params(axis='y')
    
    # Second y-axis for orientation
    ax2 = ax_combined.twinx()
    sns.boxplot(data=orientation_data, x='Case', y='Orientation', ax=ax2, 
                color="#FF8C89", positions=positions+width/2, gap=0.48)
    ax2.set_ylabel('Orientation [°]', color=orientation_color)
    ax2.tick_params(axis='y')
    
    # Adjust x-axis
    ax_combined.set_xticks(positions)
    ax_combined.set_xticklabels(['1', '2', '3', '4'])

    plt.tight_layout()
    # plt.savefig('rrtstar_apf_analysis.pdf', format='pdf', dpi=150, bbox_inches='tight')
    plt.show()


paths_data = read_traverse_rm_files()
images = read_images_from_directory()
all_obstacles = process_data(paths_data, images)

analyze_data(paths_data, all_obstacles)
