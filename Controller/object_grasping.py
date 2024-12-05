import sys
sys.path.append('D:/Robot 2SR/2sr-swarm-control')
from Model import global_var as gv, robot2sr as rsr, manipulandum, splines
import motive_client, robot2sr_controller as rsr_ctrl, camera_optitrack_synchronizer as cos
import threading
from datetime import datetime
import time
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import nearest_points
from shapely import affinity
import cv2
import numpy as np
import cvxpy
import os
import json
from typing import List
from scipy.optimize import minimize
from skopt import gp_minimize
from scipy.spatial import Voronoi
from scipy.signal import find_peaks
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as PlotPolygon
from matplotlib.collections import PatchCollection
from matplotlib import animation
from collections import deque
from scipy.optimize import minimize
import pickle

agent: rsr.Robot = None
object: manipulandum.Shape = None

mocap = motive_client.MocapReader()
rgb_camera = cos.Aligner()
agent_controller = rsr_ctrl.Controller()

markers = None
agent_width = 0.07
agent_length = 0.33

original_obstacles = []
expanded_obstacles = []
grasp_pose = None
target_pose = None

workspace_bounds = [-0.606, 0.319, -1.02, 1.034]

TARGET_SPEED = 0.06
lookahead_distance = 0.07

T = 20
NX = 3
NU = 3

R = np.diag([10000, 0.5, 0.002]) # input cost matrix 
Q = np.diag([10, 10, 0.0]) # cost matrixq
Qf = Q # final matrix
Rd = np.diag([10, 10000, 0.001])

directory = "Experiments/Data/Tracking/Grasp"
date_title = None
filename = None

tracking_data = []
path_data = []
elapsed_time = 0
start_time = None

# simulation = True
simulation = False


# ----------------------------- State Estimation ----------------------------

def updateConfig() -> None:
    global agent, object, markers

    agents_config, objects_config, markers, msg = mocap.getConfig()

    if agents_config and objects_config:
        agent_config = agents_config[0]
        if agent:
            if not simulation:
                agent.pose = [agent_config['x'], agent_config['y'], agent_config['theta']]
                agent.k1 = agent_config['k1']
                agent.k2= agent_config['k2']
                agent.head.pose = agent_config['head']
                agent.tail.pose = agent_config['tail']
        else:
            agent = rsr.Robot(agent_config['id'], agent_config['x'], agent_config['y'], agent_config['theta'], agent_config['k1'], agent_config['k2'])

        object_config = objects_config[0]
        object_pose = [object_config['x'], object_config['y'], object_config['theta']]
        if object:
            if not simulation:
                object.pose = object_pose
            else:
                pass
        else:
            object = manipulandum.Shape(object_config['id'], object_pose)
    else:
        print(msg)

def updateConfigLoop() -> None:
    while True:
        updateConfig()

        rgb_camera.markers = markers
        if object is not None:
            rgb_camera.manip_center = object.pose

def expandObstacles() -> None:
    global original_obstacles, expanded_obstacles

    obstacles_corners = rgb_camera.obstacles
    
    contour = object.contour.T
    # Calculate perimeter
    perimeter = cv2.arcLength(contour.astype(np.float32), closed=True)
    
    # Approximate polygon
    epsilon = 0.01 * perimeter
    approx = cv2.approxPolyDP(contour.astype(np.float32), epsilon, closed=True)

    obstacles_corners.append(approx.squeeze())

    for corners in obstacles_corners:

        obstacle_poly = Polygon(corners)
        original_obstacles.append(obstacle_poly)
        
        # For rectangular robot, we need to consider worst-case expansion
        # This is the radius of the circle that contains the robot
        # worst_case_radius = np.sqrt((agent_length/2)**2 + (agent_width/2)**2)
        worst_case_radius = 0.02
        expanded_poly = obstacle_poly.buffer(worst_case_radius, join_style=2)
        expanded_obstacles.append(expanded_poly)

        extended_corners = []
        xx, yy = expanded_poly.exterior.coords.xy
        for x, y in zip(xx.tolist()[:-1], yy.tolist()[:-1]):
            extended_corners.append((x, y))
        rgb_camera.expanded_obstacles_global.append(extended_corners)

def setUpEnv() -> None:
    global grasp_pose
    # ------------------------ Start tracking -----------------------
    '''
    Detect locations and geometries of:
        - robot
        - object to grasp
        - obstacles
    '''

    print('Start Motive streaming...')
    mocap.startDataListener() 

    update_thread = threading.Thread(target=updateConfigLoop)
    update_thread.daemon = True  
    update_thread.start()

    while not agent or not object:
        pass

    date_title = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = f"{directory}/obtacles_heart_{date_title}.json"
    rgb_camera.startVideo(date_title, task='object_grasp')

    print('Waiting for the video to start...')
    while not rgb_camera.wait_video:
        pass

    print('Video started')
    print()
    # ---------------------------------------------------------------

    # ------------------ Create the environment map -----------------
    '''
    Steps:
        - transform obstacles by expanding them based on robot's 
        geometry (Minkowski sum)
    '''
    # ---------------------------------------------------------------
    
    # ---------------- Determine grasping parameters ----------------
    '''
    Options:
        - grasp from the side opposite to the heading direction
        - optimization approach based on minimum forces and shape

    !! Estimate the grasp feasibility with given obstacles
    '''

    object.delta_theta = np.pi/2 - object.theta
    rgb_camera.direction = object.heading_angle

    dir = object.heading_angle - np.pi
    direction_vector = np.array([np.cos(dir), np.sin(dir)])

    # Find a point on the contour in the opposite direction
    s_array = np.linspace(0, 1, 200)
    max_dot_product = 0
    margin_in = 0.02
    margin_out = 0.08

    for s in s_array:
        point = object.getPoint(s)
        theta = object.getTangent(s)

        vector_to_point = np.array(point) - object.position
        dot_product = np.dot(vector_to_point, direction_vector)
        
        if dot_product > max_dot_product:
            max_dot_product = dot_product

            point_with_margin_out = [point[0] + margin_out * np.cos(dir), 
                                    point[1] + margin_out * np.sin(dir)]
            target_pose = point_with_margin_out + [normalizeAngle(theta)]

            point_with_margin_in = [point[0] + margin_in * np.cos(dir), 
                                    point[1] + margin_in * np.sin(dir)]
            grasp_pose = point_with_margin_in + [normalizeAngle(theta)]

    rgb_camera.grasp_point = grasp_pose[:2]  # Store the grasp point
    # ---------------------------------------------------------------

    # ------------------------ Path planning ------------------------
    '''
    ** Start first with path planning that does not require reshaping 
    of the robot 
    '''

    while True:
        if rgb_camera.obstacles is not None:
            break

    expandObstacles()

    # # runRigidPlanner()

    # # Save the current path points to the file
    # # run Voronoi analysis
    # analyzer, initial_pose = runVoronoi()
    # # plt.show()

    # # Update the agent's pose
    # if simulation:
    #     agent.pose = initial_pose
    
    # # Find passage sequence
    # print('Find passage route...')
    # print()
    # rear_path_points, front_path_points, middle_path_points = analyzer.find_passage_sequence(agent.pose, target_pose, agent_length)

    # rear_path_points = [rear_path_points_i.tolist() for rear_path_points_i in rear_path_points]
    # front_path_points = [front_path_points_i.tolist() for front_path_points_i in front_path_points]
    # middle_path_points = [middle_path_points_i.tolist() for middle_path_points_i in middle_path_points]

    # savePathPoints(rear_path_points, front_path_points, middle_path_points)

    # return rear_path_points, front_path_points, middle_path_points

# ----------------------------- Static Functions ----------------------------

def normalizeAngle(angle: float) -> float:
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

def close2Goal(current: list, target: list) -> bool:
    status = True

    # Calculate Euclidean distance between current and target (x, y)
    distance = np.linalg.norm(np.array(current) - np.array(target))
    
    # Define thresholds for position and orientation
    distance_threshold = 0.02

    if distance > distance_threshold:
        status = False
    
    print(f"Distance to goal: {distance:.3f} m")
    print()

    return status

def close2Shape(current_k: list, target_k: list) -> bool:
    status = True

    k1_diff = abs(current_k[0] - target_k[0])
    k2_diff = abs(current_k[1] - target_k[1])
    print(f'k1 diff: {k1_diff}')
    print(f'k2 diff: {k2_diff}')

    if k1_diff > 5.5 or k2_diff > 5.5:
        status = False
    return status

# --------------------------- Rigid Motion Planning -------------------------

class Node:
    def __init__(self, position, theta=0.0):
        self.position = position  # [x, y]
        self.theta = theta
        self.parent = None
        self.cost = 0.0

    @property
    def pose(self):
        return [*self.position, self.theta]

class RRTStar:
    def __init__(self, start_pose, target_pose, bounds, max_dth=2*np.pi, step_size=0.2, 
                 max_iter=1000, search_radius=1.0, w1=10.0, w2=0.1, w3=200):
        self.start = Node(np.array(start_pose[:-1]), start_pose[-1])
        self.goal = Node(np.array(target_pose[:-1]), target_pose[-1])
        self.bounds = bounds  # [x_min, x_max, y_min, y_max]
        self.max_dth = max_dth
        self.max_iter = max_iter
        self.step_size = step_size
        self.search_radius = search_radius
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.nodes = [self.start]

    def getRobotPolygon(self, pose):
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

    def plan(self):
        for i in range(self.max_iter):
            # Sample random pose
            if np.random.random() < 0.05:  # 5% chance to sample goal
                sampled_pos = self.goal.position
            else:
                sampled_pos = self.sampleRandomPosition()
            
            # Find nearest node
            nearest_node = self.getNearestNode(sampled_pos)
            
            # Extend towards sampled position
            new_pos = self.steer(nearest_node.position, sampled_pos)
            rgb_camera.new_pos = new_pos

            # Find optimal theta for new position
            optimal_theta = self.findOptimalTheta(new_pos, nearest_node)

            if optimal_theta is None:
                continue  # Skip if no valid theta found
            
            # Check if new pose is valid
            if (self.isValidPose(new_pos, optimal_theta) and 
                self.isPathValid(nearest_node, new_pos, optimal_theta)):
                
                new_node = Node(new_pos, optimal_theta)

                # Find nearby nodes for rewiring
                nearby_nodes = self.getNearbyNodes(new_pos)

                # Choose best parent from nearby nodes
                best_parent, cost = self.chooseBestParent(new_node, nearby_nodes)

                if best_parent is not None:
                    new_node.parent = best_parent
                    new_node.cost = cost
                    self.nodes.append(new_node)
                    
                    # Rewire nearby nodes
                    self.rewire(new_node, nearby_nodes)
                    rgb_camera.all_nodes = self.nodes
                    
                    # time.sleep(0.5)
                    
                    # Check if we can connect to goal
                    distance_to_goal = np.linalg.norm(new_pos - self.goal.position)
                    dth = abs(self.goal.theta - new_node.theta) / distance_to_goal
                    
                    # if (distance_to_goal < self.step_size and dth <= self.max_dth and
                    #     self.isPathValid(new_node, self.goal.position, self.goal.theta)):
                    if (distance_to_goal < self.step_size and
                        self.isPathValid(new_node, self.goal.position, self.goal.theta)):
                        if dth <= self.max_dth:
                            self.goal.parent = new_node
                            self.goal.cost = self.calculateCost(new_node, self.goal)
                            rgb_camera.all_nodes.append(self.goal)
                            return self.extractPath()
                        else:
                            continue
        
        return None  # No path found
    
    def sampleRandomPosition(self):
        x = np.random.uniform(self.bounds[0], self.bounds[1])
        y = np.random.uniform(self.bounds[2], self.bounds[3])
        return np.array([x, y])
    
    def findOptimalTheta(self, position, from_node: Node):
        def objective(theta):
            # Get robot polygon at proposed position and orientation
            robot_polygon = self.getRobotPolygon([position[0], position[1], theta[0]])
            
            # Calculate minimum distance to obstacles
            min_distance = float('inf')
            for obstacle in expanded_obstacles:
                distance = robot_polygon.distance(obstacle)
                min_distance = min(min_distance, distance)
            
            # Three terms to consider:
            # 1. Distance from obstacles
            # 2. Change from previous theta
            # 3. Difference from goal theta
            dist = np.linalg.norm(position - from_node.position)
            theta_change = abs(theta[0] - from_node.theta)
            goal_theta_diff = abs(theta[0] - self.goal.theta)
            
            # Weight factors can be tuned
            # return -self.w1 * min_distance + self.w2 * theta_change + np.exp(-self.w3 * dist) * goal_theta_diff
            return -self.w1 * min_distance + self.w2 * theta_change


        # Initial guess is current theta
        theta0 = [from_node.theta]
        bounds = [(-np.pi, np.pi)]
        
        # Optimize
        result = minimize(objective, theta0, bounds=bounds, method='SLSQP')
        
        if result.success:
            optimal_theta = result.x[0]
            # Verify the solution is actually valid
            distance = np.linalg.norm(position - from_node.position)
            dth = abs(optimal_theta - from_node.theta) / distance

            if dth > self.max_dth:
                optimal_theta = from_node.theta + np.sign(optimal_theta) * distance * self.max_dth

            return optimal_theta
        
        return None
    
    def getNearestNode(self, position):
        distances = [np.linalg.norm(node.position - position) for node in self.nodes]
        return self.nodes[np.argmin(distances)]
    
    def getNearbyNodes(self, position):
        nearby = []
        for node in self.nodes:
            if self.calculateDistance(node.position, position) <= self.search_radius:
                nearby.append(node)
        return nearby
    
    def calculateCost(self, node_parent: Node, node: Node):
        # Include both distance and orientation change in cost
        distance_cost = np.linalg.norm(node.position - node_parent.position)
        angle_cost = abs(node.theta - node_parent.theta)
        return node_parent.cost + distance_cost + 0.3 * angle_cost  # Weight can be adjusted
    
    def chooseBestParent(self, new_node: Node, nearby_nodes: List[Node]):
        min_cost = float('inf')
        best_parent = None
        
        for node in nearby_nodes:
            # Calculate potential cost through this node
            potential_cost = self.calculateCost(node, new_node)
            
            # Check if this path is valid and better than current best
            if (self.isPathValid(node, new_node.position, new_node.theta) and 
                potential_cost < min_cost):
                min_cost = potential_cost
                best_parent = node
                
        return best_parent, min_cost
    
    def steer(self, from_pos, to_pos):
        direction = to_pos - from_pos
        distance = np.linalg.norm(direction)
        if distance > self.step_size:
            direction = direction / distance * self.step_size
        return from_pos + direction
    
    def rewire(self, new_node: Node, nearby_nodes: List[Node]):
        for node in nearby_nodes:
            if node == new_node.parent:
                continue
            
            potential_cost = self.calculateCost(new_node, node)
            
            if (potential_cost < node.cost and 
                self.isPathValid(new_node, node.position, node.theta)):
                node.parent = new_node
                node.cost = potential_cost
    
    def calculateDistance(self, pose1, pose2):
        # Euclidean distance for position, weighted angular difference
        pos_diff = np.sqrt((pose1[0] - pose2[0])**2 + (pose1[1] - pose2[1])**2)
        # angle_diff = abs(normalizeAngle(pose1[2] - pose2[2]))
        # return pos_diff + 0.2 * angle_diff  # Weight for angular component
        return pos_diff
    
    def isValidPose(self, position, theta):
        # Check bounds
        if (position[0] < self.bounds[0] or position[0] > self.bounds[1] or
            position[1] < self.bounds[2] or position[1] > self.bounds[3]):
            return False
        
        # Check robot collision
        robot_polygon = self.getRobotPolygon([position[0], position[1], theta])
        if any(obs.intersects(robot_polygon) for obs in expanded_obstacles):
            return False
        return True
    
    def isPathValid(self, from_node: Node, to_pos, to_theta):
        # Check straight line path
        line = LineString([(from_node.position[0], from_node.position[1]), 
                          (to_pos[0], to_pos[1])])
        
        # Check collision at intermediate points
        num_checks = 10
        for i in range(num_checks):
            t = i / (num_checks - 1)
            point = line.interpolate(t, normalized=True)
            # Interpolate theta
            theta = from_node.theta + t * (to_theta - from_node.theta)
            
            if not self.isValidPose(np.array([point.x, point.y]), theta):
                return False
        return True

    def extractPath(self):
        if self.goal.parent is None:
            return None
        
        path = []
        node = self.goal
        while node is not None:
            path.append(node.pose)
            node = node.parent
        return path[::-1]  # Reverse path to get start-to-goal order

class RRT:
    def __init__(self, start_pose, target_pose, bounds, max_dth=np.pi, step_size=0.2, max_iter=1000,
                 w1=1.476, w2=0.34, w3=0.633):
        self.start = Node(np.array(start_pose[:-1]), start_pose[-1])
        self.goal = Node(np.array(target_pose[:-1]), target_pose[-1])
        self.bounds = bounds # [x_min, x_max, y_min, y_max]
        self.max_dth = max_dth
        self.step_size = step_size
        self.max_iter = max_iter
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.nodes = [self.start]

    def getRobotPolygon(self, pose):
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
        
    def sampleRandomPosition(self):
        x = np.random.uniform(self.bounds[0], self.bounds[1])
        y = np.random.uniform(self.bounds[2], self.bounds[3])
        return np.array([x, y])
    
    def findOptimalTheta(self, position, from_node: Node):
        def objective(theta):
            # Get robot polygon at proposed position and orientation
            robot_polygon = self.getRobotPolygon([position[0], position[1], theta[0]])
            
            # Calculate minimum distance to obstacles
            min_distance = float('inf')
            for obstacle in expanded_obstacles:
                distance = robot_polygon.distance(obstacle)
                min_distance = min(min_distance, distance)
            
            # Three terms to consider:
            # 1. Distance from obstacles
            # 2. Change from previous theta
            # 3. Difference from goal theta
            theta_change = abs(theta[0] - from_node.theta)
            goal_theta_diff = abs(theta[0] - self.goal.theta)
            
            # Weight factors can be tuned
            return -self.w1 * min_distance + self.w2 * theta_change + self.w3 * goal_theta_diff

        # Initial guess is current theta
        theta0 = [from_node.theta]
        bounds = [(-np.pi, np.pi)]
        
        # Optimize
        result = minimize(objective, theta0, bounds=bounds, method='SLSQP')
        
        if result.success:
            optimal_theta = result.x[0]
            # Verify the solution is actually valid
            distance = np.linalg.norm(position - from_node.position)
            dth = abs(optimal_theta - from_node.theta) / distance

            robot_polygon = self.getRobotPolygon([position[0], position[1], optimal_theta])
            if (not any(obs.intersects(robot_polygon) for obs in expanded_obstacles)
                and dth <= self.max_dth):
                return optimal_theta
        
        return None

    def getNearestNode(self, position):
        distances = [np.linalg.norm(node.position - position) for node in self.nodes]
        return self.nodes[np.argmin(distances)]

    def steer(self, from_pos, to_pos):
        direction = to_pos - from_pos
        distance = np.linalg.norm(direction)
        if distance > self.step_size:
            direction = direction / distance * self.step_size
        return from_pos + direction

    def isValidPosition(self, position, theta):
        # Check bounds
        if (position[0] < self.bounds[0] or position[0] > self.bounds[1] or
            position[1] < self.bounds[2] or position[1] > self.bounds[3]):
            return False
        
        # Check robot collision
        robot_polygon = self.getRobotPolygon([position[0], position[1], theta])
        if any(obs.intersects(robot_polygon) for obs in expanded_obstacles):
            return False
        return True

    def isPathValid(self, from_node: Node, to_pos, to_theta):
        # Check straight line path
        line = LineString([(from_node.position[0], from_node.position[1]), 
                          (to_pos[0], to_pos[1])])
        
        # Check collision at intermediate points
        num_checks = 10
        for i in range(num_checks):
            t = i / (num_checks - 1)
            point = line.interpolate(t, normalized=True)
            # Interpolate theta
            theta = from_node.theta + t * (to_theta - from_node.theta)
            
            if not self.isValidPosition(np.array([point.x, point.y]), theta):
                return False
        return True

    def extractPath(self):
        if self.goal.parent is None:
            return None
        
        path = []
        node = self.goal
        while node is not None:
            path.append(node.pose)
            node = node.parent
        return path[::-1]  # Reverse path to get start-to-goal order

    def plan(self):
        for i in range(self.max_iter):
            # Sample random position
            if np.random.random() < 0.05:  # 5% chance to sample goal
                sampled_pos = self.goal.position
            else:
                sampled_pos = self.sampleRandomPosition()
            
            # Find nearest node
            nearest_node = self.getNearestNode(sampled_pos)
            
            # Extend towards sampled position
            new_pos = self.steer(nearest_node.position, sampled_pos)
            rgb_camera.new_pos = new_pos

            # Find optimal theta for new position
            optimal_theta = self.findOptimalTheta(new_pos, nearest_node)

            if optimal_theta is None:
                continue  # Skip if no valid theta found
            
            # Check if new position is valid
            if (self.isValidPosition(new_pos, optimal_theta) and 
                self.isPathValid(nearest_node, new_pos, optimal_theta)):
                
                new_node = Node(new_pos, optimal_theta)
                new_node.parent = nearest_node
                self.nodes.append(new_node)

                rgb_camera.all_nodes = self.nodes
                rgb_camera.new_pos = new_pos
                # time.sleep(0.5)
                
                # Check if we can connect to goal
                distance_to_goal = np.linalg.norm(new_pos - self.goal.position)
                dth = abs(self.goal.theta - new_node.theta) / distance_to_goal
                
                if (distance_to_goal < self.step_size and dth <= self.max_dth and
                    self.isPathValid(new_node, self.goal.position, self.goal.theta)):
                    self.goal.parent = new_node
                    rgb_camera.all_nodes.append(self.goal)
                    return self.extractPath()
        
        return None  # No path found

class BayesianTuner:
    def __init__(self, workspace_bounds, start_pose, target_pose):
        self.workspace_bounds = workspace_bounds
        self.start_pose = start_pose
        self.target_pose = target_pose
        
        # Define parameter space
        self.space = [
            (0.5, 2.0),  # w1 range
            (0.05, 1.0),  # w2 range
            (0.05, 1.0)   # w3 range
        ]
        
        # Calculate maximum possible length and clearance for normalization
        # self.max_possible_length = np.linalg.norm(
        #     np.array([workspace_bounds[1][0] - workspace_bounds[0][0],
        #              workspace_bounds[1][1] - workspace_bounds[0][1]]))
        self.max_possible_length = np.linalg.norm(
            np.array([workspace_bounds[1] - workspace_bounds[0],
                     workspace_bounds[3] - workspace_bounds[2]]))
        
        # Maximum possible clearance would be half the smallest workspace dimension
        # self.max_possible_clearance = min(
        #     workspace_bounds[1][0] - workspace_bounds[0][0],
        #     workspace_bounds[1][1] - workspace_bounds[0][1]) / 2
        self.max_possible_clearance = min(
            workspace_bounds[1] - workspace_bounds[0],
            workspace_bounds[3] - workspace_bounds[2]) / 2
        
    def getRobotPolygon(self, pose):
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

    def evaluate_path(self, path):
        if path is None:
            # return float('inf'), float('inf'), float('inf'), False
            return 10e7, 10e7, 10e7, False
        
        # Calculate metrics
        path_length = 0
        min_clearance = float('inf')
        total_orientation_change = 0
        
        for i in range(len(path)-1):
            # Path length
            path_length += np.linalg.norm(
                np.array(path[i][:2]) - np.array(path[i+1][:2]))
            
            # Minimum clearance to obstacles
            robot_polygon = self.getRobotPolygon(path[i])
            for obs in expanded_obstacles:
                clearance = robot_polygon.distance(obs)
                min_clearance = min(min_clearance, clearance)
            
            # Smoothness of orientation changes
            orientation_change = abs(path[i][2] - path[i+1][2])
            total_orientation_change += orientation_change

        return path_length, min_clearance, total_orientation_change, True

    def objective_score(self, metrics):
        path_length, min_clearance, orientation_change, success = metrics
        if not success:
            # return float('inf')
            return 10e7
            
        # Normalize metrics
        norm_length = path_length / self.max_possible_length
        norm_clearance = 1.0 - (min_clearance / self.max_possible_clearance)
        norm_orientation = orientation_change / (2 * np.pi)
        
        # Combined score (lower is better)
        return (0.4 * norm_length + 
                0.3 * norm_clearance + 
                0.3 * norm_orientation)

    def objective(self, params):
        w1, w2, w3 = params
        
        # Run multiple trials
        trial_metrics = []
        for _ in range(3):
            rrt = RRT(self.start_pose, self.target_pose, 
                      self.workspace_bounds, w1=w1, w2=w2, w3=w3)
            path = rrt.plan()
            metrics = self.evaluate_path(path)
            trial_metrics.append(metrics)
        
        # Average performance
        avg_metrics = tuple(np.mean([m[i] for m in trial_metrics]) 
                          for i in range(4))
        return self.objective_score(avg_metrics)

    def tune_parameters(self, n_calls=30):
        result = gp_minimize(self.objective,
                           self.space,
                           n_calls=n_calls,
                           n_random_starts=10,
                           noise=0.1,
                           random_state=42)
        
        return {
            'w1': result.x[0],
            'w2': result.x[1],
            'w3': result.x[2],
            'score': result.fun
        }

# --------------------------- Rigid Motion Control --------------------------

def getVelAlongHeading(v: list, angle: float) -> np.ndarray:
    # 1. Convert body velocities to global frame
    R = np.array([[np.cos(agent.theta), -np.sin(agent.theta)],
                  [np.sin(agent.theta), np.cos(agent.theta)]])
    v_body = np.array(v[:-1])
    v_global = R @ v_body
    
    # 2. Create heading direction unit vector
    heading_vector = np.array([np.cos(angle), 
                             np.sin(angle)])
    
    # 3. Project global velocity onto heading direction
    v_along_heading = np.dot(v_global, heading_vector)
    
    return v_along_heading

def calcRefTrajectory(cx: list, cy: list, cyaw: list, sp, ind, v) -> tuple[np.ndarray, np.ndarray]:
    qref = np.zeros((NX, T + 1))
    vref = np.zeros((1, T + 1))
    ncourse = len(cx)

    qref[0, 0] = cx[ind]
    qref[1, 0] = cy[ind]
    qref[2, 0] = cyaw[ind]
    vref[0, 0] = sp[ind]
    travel = 0.0

    for i in range(1, T + 1):
        travel += abs(v) * gv.DT
        dind = int(round(travel / lookahead_distance))
        if (ind + dind) < ncourse:
            qref[0, i] = cx[ind + dind]
            qref[1, i] = cy[ind + dind]
            qref[2, i] = cyaw[ind + dind]
            vref[0, i] = sp[ind + dind]
        else:
            qref[0, i] = cx[ncourse - 1]
            qref[1, i] = cy[ncourse - 1]
            qref[2, i] = cyaw[ncourse - 1]
            vref[0, i] = sp[ncourse - 1]

    return qref, vref

def getLinearModelMatrix(vref: float, phi: float) -> tuple[np.ndarray, np.ndarray]:
    A = np.zeros((NX, NX))
    A[0, 0] = 1.0
    A[0, 2] = -vref * np.sin(phi) * gv.DT
    A[1, 1] = 1.0
    A[1, 2] = vref * np.cos(phi) * gv.DT
    A[2, 2] = 1.0

    B = np.zeros((NX, NU))
    B[0, 0] = gv.DT * np.cos(phi)
    B[0, 1] = -gv.DT * np.sin(phi)
    B[1, 0] = gv.DT * np.sin(phi)
    B[1, 1] = gv.DT * np.cos(phi)
    B[2, 2] = gv.DT

    return A, B

def mpc(qref: np.ndarray, vref: np.ndarray, heading_angle: float) -> tuple[list, np.ndarray]:
    q = cvxpy.Variable((NX, T + 1))
    u = cvxpy.Variable((NU, T))

    cost = 0.0
    constraints = []

    constraints += [q[:, 0] == agent.pose - qref[:,0]]  

    for t in range(T):
        cost += cvxpy.quad_form(u[:, t], R)
        if t != 0:
            cost += cvxpy.quad_form(q[:, t], Q)        
        A, B = getLinearModelMatrix(vref[0, t], heading_angle)  

        constraints += [q[:, t + 1] == A @ q[:, t] + B @ u[:, t]]  

    cost += cvxpy.quad_form(q[:, T], Qf)  
    prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
    prob.solve(solver=cvxpy.ECOS, verbose=False)

    if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
        vx = u.value[0, 0] + vref[0, 1]
        vy = u.value[1, 0]
        omega = u.value[2, 0]

        delta_theta = heading_angle - agent.theta
        
        vel_rot = np.array([[np.cos(delta_theta), -np.sin(delta_theta)],
                            [np.sin(delta_theta), np.cos(delta_theta)]])
        v = vel_rot.dot(np.array([[vx], [vy]]))
        vx, vy = v.flatten().tolist()

        rot = np.array([[np.cos(agent.theta), -np.sin(agent.theta), 0],
                        [np.sin(agent.theta), np.cos(agent.theta), 0],
                        [0, 0, 1]])

        q_dot = rot.dot([vx, vy, omega])
        q_new = np.array(agent.pose) + q_dot * gv.DT
    else:
        print("Error: Cannot solve mpc..")
        vx, vy, omega = None, None, None
        q_new = None

    return [vx, vy, omega], q_new

def traversePath(path: list) -> None:
    global agent, tracking_data, path_data, elapsed_time, start_time

    v_r = [0.0] * 3
    heading_angle = 0
    finish = False
    rgb_camera.add2traj(agent.position)

    path_array = np.array(path)
    traj = splines.Trajectory(path_array[:,0], path_array[:,1], path_array[:,2].tolist())

    for x, y, yaw, in zip(traj.x, traj.y, traj.yaw):
        path_data.append({'x': x, 'y': y, 'yaw': yaw})

    start_time = time.perf_counter()

    cx, cy, cyaw, s = traj.params
    sp = [TARGET_SPEED] * len(cx)
    

    while True:
        if close2Goal(agent.position, traj.getPoint(-1)) or rgb_camera.finish:
            v_r = [0.0] * 3
            finish = True
        else:
            target_ind = traj.getTarget(agent.position, lookahead_distance)

            target_point = traj.getPoint(target_ind)
            vector_to_target = np.array(target_point) - np.array(agent.position)
            heading_angle = np.arctan2(vector_to_target[1], vector_to_target[0])
            rgb_camera.heading = heading_angle

            v_heading = getVelAlongHeading(v_r, heading_angle)
            qref, vref = calcRefTrajectory(cx, cy, cyaw, sp, target_ind, v_heading) 

            v_r, q = mpc(qref, vref, heading_angle)
            
        if simulation:
            agent.pose = q
        # print(agent.pose)
        rgb_camera.add2traj(agent.position)

        v = v_r + [0.0] * 2
        agent_controller.move(agent, v, [0, 0])

        current_time = time.perf_counter()
        elapsed_time = current_time - start_time

        tracking_data.append({'time': elapsed_time,
                              'config': agent.config.tolist(),
                              'stiffness': agent.stiffness,
                              'target_vel': v})

        if finish:
            break

def runRigidPlanner():
    tuner = BayesianTuner(workspace_bounds, agent.pose, target_pose)
    best_params = tuner.tune_parameters()

    print(best_params)

    while True:
        # rrt = RRT(agent.pose, target_pose, workspace_bounds)
        rrt = RRTStar(agent.pose, target_pose, workspace_bounds)
        
        path = rrt.plan()

        if path is not None:
            print("Path found!")
            path.append(grasp_pose)
            # path.append(grasp_pose[:-1])
            print(path)
            rgb_camera.rrt_path = path
            
            # Ask for user confirmation
            user_input = input("Is this path acceptable? (yes/no): ").lower()
            
            if user_input == 'yes' or user_input == 'y':
                # Continue with the original code
                interpolated_path = []
                num_interpolated_points = 20

                for i in range(len(path) - 1):
                    start = path[i]
                    end = path[i + 1]
                    for j in range(num_interpolated_points + 1):
                        t = j / num_interpolated_points
                        x = start[0] + t * (end[0] - start[0])
                        y = start[1] + t * (end[1] - start[1])
                        theta = normalizeAngle(start[2] + t * (end[2] - start[2]))
                        interpolated_path.append([x, y, theta])

                # print('Traverse the path...')
                # traversePath(interpolated_path)
                # print('Arrived!')
                # print()

                # print('Grasp the object...')
                # grasp(path[-1])
                # print(f'Finished! Recording time: {elapsed_time} seconds')
                # rgb_camera.finish = True

                # # Prepare data to be written
                # data_json = {
                #     "metadata": {
                #         "description": "Grasp through obstacles",
                #         "date": date_title
                #     },
                #     'path': path_data,
                #     "tracking": tracking_data
                # } 

                # # Write data to JSON file
                # if not simulation:
                #     with open(filename, 'w') as f:
                #         json.dump(data_json, f, indent=2)

                #     print(f"Data written to {filename}")                   
                
                break 
                
            elif user_input == 'no' or user_input == 'n':
                rgb_camera.rrt_path = None
                print("Replanning path...")
                continue  # Continue to next iteration to find a new path
                
            else:
                print("Invalid input. Please enter 'yes' or 'no'.")
                continue  # Ask for input again if invalid
                
        else:
            print("No path found!")
            # break  # Exit if no path is found

# --------------------------- Soft Motion Planning --------------------------

# Voronoi analysis
class VoronoiPassageAnalyzer:
    def __init__(self, obstacles, workspace_bounds):
        self.obstacles = obstacles
        self.workspace_bounds = workspace_bounds
        self.boundary_points = self.sample_obstacle_boundaries()
        self.vor = Voronoi(self.boundary_points)
        self.passage_graph = self.create_passage_graph()
        self.allowed_passage_graph = None
        
    def sample_obstacle_boundaries(self, sampling_density=0.1):
        boundary_points = []
        
        # Sample points from each obstacle's boundary
        for obstacle in self.obstacles:
            if isinstance(obstacle, Polygon):
                # boundary = obstacle.exterior.coords[:-1]  # Exclude last point (same as first)
                xx, yy = obstacle.exterior.coords.xy
                xx = xx.tolist()
                yy = yy.tolist()
                points = [[xx[i], yy[i]] for i in range(len(xx) - 1)]
                boundary_points.extend(points)
        
        return np.array(boundary_points)
    
    def create_passage_graph(self):
        graph = nx.Graph()
        
        # Add Voronoi vertices as graph nodes
        for i, vertex in enumerate(self.vor.vertices):
            if self.is_point_in_workspace(vertex) and self.is_point_collision_free(vertex):
                clearance = self.get_clearance(vertex)
                graph.add_node(i, pos=vertex, clearance=clearance)
        
        # Add Voronoi edges
        for ridge_vertices in self.vor.ridge_vertices:
            if -1 not in ridge_vertices:  # Skip infinite ridges
                v1, v2 = ridge_vertices
                if (v1 in graph.nodes and v2 in graph.nodes):
                    p1 = self.vor.vertices[v1]
                    p2 = self.vor.vertices[v2]
                    
                    # Check if edge intersects any obstacle
                    if self.is_line_collision_free(p1, p2):
                        edge_length = np.linalg.norm(p1 - p2)
                        min_clearance = min(
                            graph.nodes[v1]['clearance'],
                            graph.nodes[v2]['clearance']
                        )
                        
                        graph.add_edge(v1, v2, 
                                     length=edge_length,
                                     clearance=min_clearance)
        
        return graph

    def find_nearest_node(self, pos):
        closest_node = None
        min_distance = float('inf')

        for node in self.allowed_passage_graph.nodes:
            node_pos = self.allowed_passage_graph.nodes[node]['pos']
            distance = np.linalg.norm(np.array(node_pos) - np.array(pos))
            if distance < min_distance:
                min_distance = distance
                closest_node = node

        return closest_node, min_distance

    def add_new_edge(self, new_point, closest_node):
        new_node_id = max(self.allowed_passage_graph.nodes()) + 1
        self.allowed_passage_graph.add_node(new_node_id, 
                                        pos=new_point,
                                        clearance=self.get_clearance(new_point))
        
        closest_node_pos = self.allowed_passage_graph.nodes[closest_node]['pos']
        edge_length = np.linalg.norm(new_point - np.array(closest_node_pos))
        min_clearance = min(self.get_clearance(new_point),
                        self.allowed_passage_graph.nodes[closest_node]['clearance'])
        
        edge = (new_node_id, closest_node)
        self.allowed_passage_graph.add_edge(new_node_id, closest_node,
                                        length=edge_length,
                                        clearance=min_clearance)
        
        return new_node_id, edge, min_clearance, edge_length
    
    def identify_passages(self, min_clearance_threshold=0.5):
        passages = []
        self.allowed_passage_graph = nx.Graph()  # Create new graph for passages
        
        # Find connected components in the graph
        components = list(nx.connected_components(self.passage_graph))
        
        for component in components:
            subgraph = self.passage_graph.subgraph(component)
            
            # Find bottlenecks in this component
            bottlenecks = self.find_bottlenecks(subgraph, min_clearance_threshold)
            passages.extend(bottlenecks)
            
            # Add these passages to the new graph
            for passage in bottlenecks:
                node1, node2 = passage['nodes']
                self.allowed_passage_graph.add_node(node1, pos=passage['points'][0], 
                                                    clearance=self.get_clearance(passage['points'][0]))
                self.allowed_passage_graph.add_node(node2, pos=passage['points'][1], 
                                                    clearance=self.get_clearance(passage['points'][0]))
                self.allowed_passage_graph.add_edge(node1, node2,
                                                    length=passage['length'],
                                                    clearance=passage['clearance'])  # Add edge with passage data
        
        # Find open end nodes (nodes with degree 1)
        open_end_nodes = [node for node, degree in self.allowed_passage_graph.degree() if degree == 1]
        
        if not open_end_nodes:
            return passages, None
        
        # Find closest open end node to agent
        closest_node = None
        min_distance = float('inf')
        for node in open_end_nodes:
            node_pos = self.allowed_passage_graph.nodes[node]['pos']
            distance = np.linalg.norm(np.array(node_pos) - np.array(agent.position))
            if distance < min_distance:
                min_distance = distance
                closest_node = node
        
        # Get position of closest node
        closest_node_pos = self.allowed_passage_graph.nodes[closest_node]['pos']
        
        # Calculate direction vector from closest node to agent
        direction = np.array(agent.position) - np.array(closest_node_pos)
        angle = np.arctan2(direction[1], direction[0])
        direction = direction / np.linalg.norm(direction)
        
        # Find new point position
        # Start from the closest node and move towards agent until clearance becomes too small
        step_size = 0.05  # Adjust as needed
        current_pos = np.array(closest_node_pos)
        new_point = None
        
        while True:
            next_pos = current_pos + step_size * direction
            clearance = self.get_clearance(next_pos)
            
            if clearance > 1.3 * agent_length:
                new_point = current_pos  # Use last valid position
                break
            
            if np.linalg.norm(next_pos - np.array(agent.position)) < step_size:
                new_point = next_pos
                break
                
            current_pos = next_pos
        
        if new_point is not None:
            # Create new node and edge
            new_node_id, edge, clearance, edge_length = self.add_new_edge(new_point, closest_node)
            
            # Add new passage
            new_passage = {
                'points': (new_point, closest_node_pos),
                'nodes': edge,
                'clearance': clearance,
                'length': edge_length,
                'orientation': angle
            }
            passages.append(new_passage)
            
            return passages, (*new_point, angle)
        
        return passages, None
    
    def is_point_collision_free(self, point):
        """Check if point is inside any obstacle"""
        point_obj = Point(point)
        for obstacle in self.obstacles:
            if obstacle.contains(point_obj) or obstacle.boundary.contains(point_obj):
                return False
        return True

    def is_line_collision_free(self, p1, p2):
        """Check if line segment between p1 and p2 collides with any obstacle"""
        line = LineString([p1, p2])
        for obstacle in self.obstacles:
            if line.intersects(obstacle):
                return False
        return True

    def get_clearance(self, point):
        """Get minimum distance from point to any obstacle"""
        point_obj = Point(point)
        min_distance = float('inf')
        for obstacle in self.obstacles:
            distance = point_obj.distance(obstacle)
            min_distance = min(min_distance, distance)
        return min_distance

    def find_bottlenecks(self, graph, min_clearance_threshold):
        robot_width = agent_width  # replace with your robot's width
        safety_margin = 1.05
        min_clearance_threshold = (robot_width * safety_margin) / 2
        
        bottlenecks = []
        
        # Sort edges by clearance
        edges = list(graph.edges(data=True))
        edges.sort(key=lambda x: x[2]['clearance'])
        
        for edge in edges:
            v1, v2, data = edge
            p1 = graph.nodes[v1]['pos']
            p2 = graph.nodes[v2]['pos']
            
            # Check clearance at multiple points along the passage
            num_checks = 5
            is_passable = True
            
            # Only proceed if the passage doesn't intersect obstacles
            if self.is_line_collision_free(p1, p2):
                for i in range(num_checks):
                    t = i / (num_checks - 1)
                    # Interpolate point along passage
                    check_point = [
                        p1[0] * (1-t) + p2[0] * t,
                        p1[1] * (1-t) + p2[1] * t
                    ]
                    
                    # Check clearance at this point
                    clearance = self.get_clearance(check_point)
                    if clearance < min_clearance_threshold:
                        is_passable = False
                        break
                
                if is_passable:
                    passage = {
                        'points': (p1, p2),
                        'nodes': (v1, v2),  # Add nodes information
                        'clearance': data['clearance'],
                        'length': data['length'],
                        'orientation': np.arctan2(p2[1]-p1[1], p2[0]-p1[0])
                    }
                    bottlenecks.append(passage)
        
        return bottlenecks
    
    def is_point_in_workspace(self, point):
        x, y = point
        return (self.workspace_bounds[0][0] <= x <= self.workspace_bounds[1][0] and
                self.workspace_bounds[0][1] <= y <= self.workspace_bounds[1][1])
    
    def find_passage_sequence(self, start_pose, goal_pose, segment_length):
        """
        Find passage sequence for a 3-point segment
        segment_length: total length of segment (distance between endpoints)
        """
        # Calculate paths of the robot points within the passages
        rear_path, front_path, interpoltaed_rear_path_points = self._calc_robot_paths(start_pose, goal_pose, segment_length)
        
        if not rear_path or not front_path:
            return None
        
        # Initialize the queue of nodes between the rear and front points
        if rear_path[0] == front_path[0]:
            path_between_rear_front = [rear_path[0]]
        else:
            path_between_rear_front = self.find_point_path(rear_path[0]['points'][0],
                                                        front_path[0]['points'][1])
        passages_between_rear_front = deque()
        for item in path_between_rear_front:
            passages_between_rear_front.append(item['nodes'])

        # Main loop
        rear_path_points = []
        front_path_points = []
        middle_path_points = []

        front_current_idx = 0

        rpp_idx = 0

        for rear_path_point in interpoltaed_rear_path_points:
            # print(f'Index of the rear path point: {rpp_idx}')

            front_pos_new = None
            middle_pos_new = None

            rear_pos = rear_path_point['pos']

            if (rear_path_point['nodes'] != passages_between_rear_front[0] and 
                rear_path_point['nodes'][::-1] != passages_between_rear_front[0]):
                if (rear_path_point['nodes'] in passages_between_rear_front or 
                    rear_path_point['nodes'][::-1] in passages_between_rear_front):
                    passages_between_rear_front.popleft()
                else:
                    if rear_path_point['nodes'][1] == passages_between_rear_front[0][0]:
                        passages_between_rear_front.appendleft(rear_path_point['nodes'])
                    elif rear_path_point['nodes'][0] == passages_between_rear_front[0][0]:
                        passages_between_rear_front.appendleft(rear_path_point['nodes'][::-1])
                    else:
                        passages_between_rear_front.popleft()
                        if rear_path_point['nodes'][1] == passages_between_rear_front[0][0]:
                            passages_between_rear_front.appendleft(rear_path_point['nodes'])
                        elif rear_path_point['nodes'][0] == passages_between_rear_front[0][0]:
                            passages_between_rear_front.appendleft(rear_path_point['nodes'][::-1])

            while front_pos_new is None:
                delta_rear_front = 0
                delta_rear_middle = 0
                middle_passage = None

                for i in range(len(passages_between_rear_front)):
                    passage = passages_between_rear_front[i]
                    if i == 0:
                        passage_start_pos = rear_pos
                    else:
                        passage_start_pos = self.allowed_passage_graph.nodes[passage[0]]['pos']

                    delta_rear_front += np.linalg.norm(self.allowed_passage_graph.nodes[passage[1]]['pos'] - passage_start_pos)

                    if middle_passage is None:
                        delta_rear_middle = delta_rear_front
                        if delta_rear_middle >= segment_length/2:
                            middle_passage = passage
                    
                    if delta_rear_front >= segment_length:
                        if i < len(passages_between_rear_front) - 1:
                            front_current_idx -= (len(passages_between_rear_front) - 1 - i)
                        break

                if delta_rear_front >= segment_length:
                    node_start_pos = passage_start_pos
                    node_end_pos = self.allowed_passage_graph.nodes[passage[1]]['pos']
                    orientation = np.arctan2(node_end_pos[1]-node_start_pos[1], 
                                             node_end_pos[0]-node_start_pos[0]) - np.pi
                    front_pos_new = node_end_pos + (delta_rear_front-segment_length) * np.array([np.cos(orientation), np.sin(orientation)])
                
                    if middle_passage[0] is None:
                        middle_node_start_pos = rear_pos
                    else:
                        middle_node_start_pos = self.allowed_passage_graph.nodes[middle_passage[0]]['pos']
                    middle_node_end_pos = self.allowed_passage_graph.nodes[middle_passage[1]]['pos']
                    middle_orientation = np.arctan2(middle_node_end_pos[1]-middle_node_start_pos[1], 
                                                    middle_node_end_pos[0]-middle_node_start_pos[0]) - np.pi
                    middle_pos_new = middle_node_end_pos + (delta_rear_middle-segment_length/2) * np.array([np.cos(middle_orientation), np.sin(middle_orientation)])
                else:
                    nodes_to_append = None

                    if front_current_idx < len(front_path) - 1:
                        front_current_idx += 1
                        nodes = front_path[front_current_idx]['nodes']

                        if nodes[1] != None:
                            nodes_to_append = nodes
                        else:
                            for neighbor in self.allowed_passage_graph.neighbors(nodes[0]):
                                if neighbor != passages_between_rear_front[-1][0]:
                                    nodes_to_append = (nodes[0], neighbor)
                                    break
                    else:
                        neighbors = self.allowed_passage_graph.neighbors(passages_between_rear_front[-1][1])
                        neighbors_list = list(neighbors)
                        # print(neighbors_list)
                        # print('Reached the end of the front path')

                        max_degree = 0
                        chosen_neighbor = None

                        for neighbor in neighbors_list:
                            # print(neighbor)
                            if neighbor != passages_between_rear_front[-1][0]:
                                if self.allowed_passage_graph.degree(neighbor) > max_degree:
                                    max_degree = self.allowed_passage_graph.degree(neighbor)
                                    chosen_neighbor = neighbor
                        
                        if chosen_neighbor is not None:
                            nodes_to_append = (passages_between_rear_front[-1][1], chosen_neighbor)
                                
                    if nodes_to_append is not None:
                        passages_between_rear_front.append(nodes_to_append)


            rear_path_points.append(rear_pos)
            front_path_points.append(front_pos_new)
            middle_path_points.append(middle_pos_new)

            rpp_idx += 1

        return rear_path_points, front_path_points, middle_path_points
    
    def _calc_robot_paths(self, start_pose, goal_pose, segment_length):
        segment_half = segment_length / 2

        front_start = np.array(start_pose[:-1]) + segment_half * np.array([-np.cos(start_pose[-1]), -np.sin(start_pose[-1])])
        rear_start = np.array(start_pose[:-1]) + segment_half * np.array([np.cos(start_pose[-1]), np.sin(start_pose[-1])])

        # Calculate target positions for endpoints
        front_target = np.array(goal_pose[:-1]) + segment_half * np.array([-np.cos(goal_pose[-1]), -np.sin(goal_pose[-1])])
        rear_target = np.array(goal_pose[:-1]) + segment_half * np.array([np.cos(goal_pose[-1]), np.sin(goal_pose[-1])])
        
        # Find path for rear point (leading the motion)
        rear_path = self.find_point_path(rear_start, rear_target)
        front_path = self.find_point_path(front_start, front_target)

        rear_nodes = []
        for item in rear_path:
            rear_nodes.append(item['nodes'])
        print(f'Rear point nodes: {rear_nodes}')

        front_nodes = []
        for item in front_path:
            front_nodes.append(item['nodes'])
        print(f'Front point nodes: {front_nodes}')
        print()

        interpoltaed_rear_path_points = self.interpolate_path_points(rear_path, rear_target)

        return rear_path, front_path, interpoltaed_rear_path_points

    def interpolate_path_points(self, path, target_point, step_size=0.02):
        interpolated_points = []

        for i in range(len(path)):
            p1, p2 = path[i]['points']
            current_nodes = path[i]['nodes']

            # If this is the last passage, find closest point to target
            if i == len(path) - 1:
                segment_vector = np.array(p2) - np.array(p1)
                segment_length = np.linalg.norm(segment_vector)
                if segment_length > 0:
                    # Calculate projection of target point onto passage vector
                    point_vector = np.array(target_point) - np.array(p1)
                    projection = np.dot(point_vector, segment_vector) / segment_length
                    
                    # Clamp projection to segment length
                    projection = min(segment_length, max(0, projection))
                    
                    # Find closest point on segment
                    closest_point = np.array(p1) + (projection / segment_length) * segment_vector
                    passage_vector = closest_point - np.array(p1)
            else:
                passage_vector = np.array(p2) - np.array(p1)
                
            passage_length = np.linalg.norm(passage_vector)
            
            if passage_length == 0:
                direction = 0
            else:
                direction = passage_vector / passage_length
            
            # How many points we need on this passage
            if passage_length < step_size:
                num_points = 1
            else:
                num_points = int(passage_length / step_size)

            for j in range(num_points):
                point_pos = np.array(p1) + j * step_size * direction                
                interpolated_points.append({
                    'pos': point_pos,
                    'nodes': current_nodes
                })

            if i == len(path) - 1 and passage_length != 0:
                interpolated_points.append({
                    'pos': closest_point,
                    'nodes': current_nodes
                })

        return interpolated_points
    
    def find_nearest_passage(self, point, goal=False):
        """Find nearest passage to a point"""
        point_array = np.array(point)
        nearest_passage = None
        min_distance = float('inf')
        
        # Check each edge in the passage graph
        for edge in self.allowed_passage_graph.edges():
            v1, v2 = edge

            if goal:
                if self.allowed_passage_graph.degree(v1) == 1 or self.allowed_passage_graph.degree(v2) == 1:
                    continue

            p1 = np.array(self.allowed_passage_graph.nodes[v1]['pos'])
            p2 = np.array(self.allowed_passage_graph.nodes[v2]['pos'])
            
            # Find closest point on line segment
            segment_vector = p2 - p1
            segment_length = np.linalg.norm(segment_vector)
            if segment_length == 0:
                continue
                
            segment_unit = segment_vector / segment_length
            point_vector = point_array - p1
            projection = np.dot(point_vector, segment_unit)
            
            if projection <= 0:
                closest_point = p1
            elif projection >= segment_length:
                closest_point = p2
            else:
                closest_point = p1 + projection * segment_unit
                
            distance = np.linalg.norm(point_array - closest_point)
            
            if distance < min_distance:
                min_distance = distance
                nearest_passage = {
                    'points': (p1, p2),
                    'nodes': edge,
                    'distance': distance,
                    'clearance': self.allowed_passage_graph.edges[edge].get('clearance', 0)
                }
        
        return nearest_passage
        
    def find_point_path(self, start_point, goal_point):
        """Find path for a point through passages"""
        start_passage = self.find_nearest_passage(start_point)
        goal_passage = self.find_nearest_passage(goal_point, goal=True)
        
        if not start_passage or not goal_passage:
            return None
            
        # Get nodes from passages
        start_nodes = start_passage['nodes']
        goal_nodes = goal_passage['nodes']  

        # print(f'Start point: {start_point}, start nodes: {start_nodes}')
        # print(f'Goal point: {goal_point}, goal nodes: {goal_nodes}')
        # print()

        # Try to find path between any combination of start and goal nodes
        shortest_path = None
        min_length = float('inf')
        
        for start_node in start_nodes:
            for goal_node in goal_nodes:
                try:
                    # Use NetworkX to find shortest path
                    path = nx.shortest_path(self.allowed_passage_graph, start_node, goal_node)
                    
                    # Calculate path length
                    path_length = 0
                    for i in range(len(path)-1):
                        n1, n2 = path[i], path[i+1]
                        p1 = np.array(self.allowed_passage_graph.nodes[n1]['pos'])
                        p2 = np.array(self.allowed_passage_graph.nodes[n2]['pos'])
                        path_length += np.linalg.norm(p2 - p1)
                    
                    if path_length < min_length:
                        min_length = path_length
                        shortest_path = path
                        
                except nx.NetworkXNoPath:
                    continue
        
        if not shortest_path:
            return None
            
        # Convert node path to passage sequence
        passage_sequence = []

        # Add initial passage
        first_node = shortest_path[0]
        first_pos = self.allowed_passage_graph.nodes[first_node]['pos']

        zero_node = next(node for node in start_nodes if node != first_node)
        zero_pos = self.allowed_passage_graph.nodes[zero_node]['pos']
        
        passage_sequence.append({
            'points': (zero_pos, first_pos),
            'nodes': (zero_node, first_node),
            'clearance': start_passage['clearance']
        })
        
        # Add passages between nodes
        for i in range(len(shortest_path)-1):
            n1, n2 = shortest_path[i], shortest_path[i+1]
            p1 = self.allowed_passage_graph.nodes[n1]['pos']
            p2 = self.allowed_passage_graph.nodes[n2]['pos']
            clearance = self.allowed_passage_graph.edges[(n1, n2)]['clearance']
            
            passage_sequence.append({
                'points': (p1, p2),
                'nodes': (n1, n2),
                'clearance': clearance
            })

        # Add final passage 
        last_node = shortest_path[-1]
        last_pos = self.allowed_passage_graph.nodes[last_node]['pos']

        goal_node = next(node for node in goal_nodes if node != last_node)
        goal_pos = self.allowed_passage_graph.nodes[goal_node]['pos']

        passage_sequence.append({
            'points': (last_pos, goal_pos),
            'nodes': (last_node, goal_node),
            'clearance': goal_passage['clearance']
        })
        
        return passage_sequence

    def find_nearest_passage_point(self, point):
        """Find nearest valid point on passage network"""
        point_array = np.array(point)
        
        # Calculate distances to all vertices
        valid_connections = []
        
        for vertex_id in self.passage_graph.nodes():
            vertex_pos = self.passage_graph.nodes[vertex_id]['pos']
            
            # Check if connection is collision-free
            if self.is_line_collision_free(point, vertex_pos):
                distance = np.linalg.norm(point_array - vertex_pos)
                clearance = min(
                    self.get_clearance(point),
                    self.get_clearance(vertex_pos)
                )
                
                valid_connections.append({
                    'node_id': vertex_id,
                    'distance': distance,
                    'clearance': clearance
                })
        
        if not valid_connections:
            return None
            
        # Return the nearest valid connection
        return min(valid_connections, key=lambda x: x['distance'])
    
    def connect_point_to_graph(self, graph, point, node_id):
        """Connect a point to nearby Voronoi vertices"""
        point_array = np.array(point)
        
        # Find closest vertices
        distances = []
        for vertex_id in graph.nodes():
            if vertex_id not in [node_id, node_id+1]:  # Skip start/goal nodes
                vertex_pos = graph.nodes[vertex_id]['pos']
                dist = np.linalg.norm(point_array - vertex_pos)
                distances.append((dist, vertex_id))
        
        # Connect to closest vertices
        distances.sort()
        for dist, vertex_id in distances[:3]:  # Connect to 3 closest vertices
            graph.add_node(node_id, pos=point)
            graph.add_edge(node_id, vertex_id, 
                         length=dist,
                         clearance=self.get_clearance(point))

class VoronoiVisualizer:
    def __init__(self, obstacles, workspace_bounds):
        self.obstacles = obstacles
        self.workspace_bounds = workspace_bounds
        self.vor = None
        self.fig = None
        self.ax = None

        self.edge_color = '#acb1c4'
        self.passage_color = '#79addc'
        
    def plot_voronoi(self, show_points=True):
        # Create figure and axis
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        
        # Plot obstacles
        self._plot_obstacles()
        
        # Sample points from obstacle boundaries
        boundary_points = self._sample_obstacle_boundaries()
        
        # Create and plot Voronoi diagram
        self.vor = Voronoi(boundary_points)
        self._plot_voronoi_diagram(show_points)
        
        # Set workspace bounds
        self.ax.set_xlim(self.workspace_bounds[0][0], self.workspace_bounds[1][0])
        self.ax.set_ylim(self.workspace_bounds[0][1], self.workspace_bounds[1][1])
        
        # Set equal aspect ratio
        self.ax.set_aspect('equal')
        
        # Add grid
        self.ax.grid(True)
        
        return self.fig, self.ax
    
    def _plot_obstacles(self):
        patches = []
        for obstacle in self.obstacles:
            if isinstance(obstacle, Polygon):
                patches.append(PlotPolygon(np.array(obstacle.exterior.coords)))
        
        # Add obstacle patches to plot
        p = PatchCollection(patches, alpha=0.4, color='gray')
        self.ax.add_collection(p)
    
    def _sample_obstacle_boundaries(self, sampling_density=0.1):
        boundary_points = []
        
        for obstacle in self.obstacles:
            if isinstance(obstacle, Polygon):
                # boundary = obstacle.exterior.coords[:-1]
                # perimeter = obstacle.length
                # num_samples = max(4, int(perimeter * sampling_density))
                
                # for i in range(num_samples):
                #     t = i / num_samples
                #     point = obstacle.boundary.interpolate(t * perimeter)
                #     boundary_points.append([point.x, point.y])
                xx, yy = obstacle.exterior.coords.xy
                xx = xx.tolist()
                yy = yy.tolist()
                points = [[xx[i], yy[i]] for i in range(len(xx) - 1)]
                boundary_points.extend(points)

        boundary_points.extend([
            [self.workspace_bounds[0][0], self.workspace_bounds[0][1]],  # bottom-left
            [self.workspace_bounds[1][0], self.workspace_bounds[0][1]],  # bottom-right
            [self.workspace_bounds[1][0], self.workspace_bounds[1][1]],  # top-right
            [self.workspace_bounds[0][0], self.workspace_bounds[1][1]]   # top-left
        ])
        
        return np.array(boundary_points)
    
    def _plot_voronoi_diagram(self, show_points=True):
        # Plot Voronoi vertices
        if show_points:
            self.ax.plot(self.vor.points[:, 0], self.vor.points[:, 1], 'ko', 
                        markersize=2, label='Obstacles Points')
        
        # Plot finite Voronoi edges
        for simplex in self.vor.ridge_vertices:
            if -1 not in simplex:
                self.ax.plot(self.vor.vertices[simplex, 0], 
                           self.vor.vertices[simplex, 1], 
                           self.edge_color, linewidth=1)
        
        # Plot infinite Voronoi edges
        center = self.vor.points.mean(axis=0)
        for pointidx, simplex in zip(self.vor.ridge_points, self.vor.ridge_vertices):
            if -1 in simplex:
                i = simplex.index(-1)
                v1 = self.vor.vertices[simplex[(i+1) % 2]]
                v2 = self.vor.points[pointidx[(i+1) % 2]] - self.vor.points[pointidx[i]]
                v2 = v2 / np.linalg.norm(v2)
                far_point = v1 + v2 * 100
                self.ax.plot([v1[0], far_point[0]], [v1[1], far_point[1]], 
                           self.edge_color, linestyle='dashed', linewidth=1)

    def highlight_narrow_passages(self, passages, linewidth=2):
        """Highlight identified narrow passages"""
        # Plot first passage with label
        offset = -3
        if passages:
            p1, p2 = passages[0]['points']
            node1, node2 = passages[0]['nodes']
            self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                        color=self.passage_color, linewidth=linewidth, 
                        label='Passages')
            
            # # Add node IDs for first passage
            # self.ax.annotate(str(node1), 
            #                 (p1[0], p1[1]),
            #                 xytext=(offset, offset),
            #                 textcoords='offset points',
            #                 fontsize=8,
            #                 color='red')
            # self.ax.annotate(str(node2), 
            #                 (p2[0], p2[1]),
            #                 xytext=(offset, offset),
            #                 textcoords='offset points',
            #                 fontsize=8,
            #                 color='red')
                
            # Plot remaining passages without labels but with node IDs
            for passage in passages[1:]:
                p1, p2 = passage['points']
                node1, node2 = passage['nodes']
                self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                            color=self.passage_color, linewidth=linewidth)
                
                # # Add node IDs
                # self.ax.annotate(str(node1), 
                #             (p1[0], p1[1]),
                #             xytext=(offset, offset),
                #             textcoords='offset points',
                #             fontsize=8,
                #             color='red')
                # self.ax.annotate(str(node2), 
                #             (p2[0], p2[1]),
                #             xytext=(offset, offset),
                #             textcoords='offset points',
                #             fontsize=8,
                #             color='red')
        
        # Add legend
        self.ax.legend()

def runVoronoi():
    global fig 

    bounds = [
        (-0.606, -0.8), # Lower left corner (x_min, y_min)
        (0.319, 0.5)  # Upper right corner (x_max, y_max)
    ]

    visualizer = VoronoiVisualizer(expanded_obstacles, bounds)
    
    # Plot basic Voronoi diagram
    fig, ax = visualizer.plot_voronoi(show_points=True)
    
    # Create analyzer and find passages
    analyzer = VoronoiPassageAnalyzer(expanded_obstacles, bounds)
    passages, initial_pose = analyzer.identify_passages()

    # passages_list = [passage['nodes'] for passage in passages]
    # print(passages_list)
    
    # Highlight narrow passages
    visualizer.highlight_narrow_passages(passages)

    with open('voronoi_plot.pkl', 'wb') as file:
        pickle.dump({'figure': fig, 'axes': ax}, file)

    return analyzer, initial_pose


# Afterprocess robot points paths
def adjustPath(path_points, angle_threshold=2, max_merge_angle=15, max_zigzag_deviation=10):
    """
    Merges segments if angle is small and removes zigzags.
    path_points: Nx2 array of (x,y) coordinates
    angle_threshold: maximum angle difference (in degrees) to consider segments parallel
    max_merge_angle: maximum angle between adjacent segments to allow merging
    max_zigzag_deviation: maximum overall direction change to consider a pattern as zigzag
    """
    path = path_points.copy()
    n_points = len(path)
    
    # Get vectors between consecutive points
    vectors = np.diff(path, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    
    # Calculate angles between consecutive segments
    dot_products = np.sum(vectors[:-1] * vectors[1:], axis=1)
    norms_product = lengths[:-1] * lengths[1:]
    angles = np.degrees(np.arccos(np.clip(dot_products / norms_product, -1.0, 1.0)))
    
    # Identify segments
    segment_starts = [0]  # First point is always a segment start
    
    # Find points where direction changes significantly
    for i in range(len(angles)):
        if abs(angles[i]) > angle_threshold:
            segment_starts.append(i + 1)
    segment_starts.append(n_points - 1)  # Add last point
    
    # Process segments
    new_path = path.copy()
    
    # First pass: merge segments with small angles
    i = 1
    while i < len(segment_starts) - 1:
        prev_start = segment_starts[i-1]
        curr_start = segment_starts[i]
        next_start = segment_starts[i + 1]
        
        # Calculate angle between previous and next segments
        prev_vector = np.array(path[curr_start]) - path[prev_start]
        next_vector = np.array(path[next_start]) - path[curr_start]
        
        # Normalize vectors
        prev_vector = prev_vector / np.linalg.norm(prev_vector)
        next_vector = next_vector / np.linalg.norm(next_vector)
        
        # Calculate angle between segments
        dot_product = np.clip(np.dot(prev_vector, next_vector), -1.0, 1.0)
        angle_between = np.degrees(np.arccos(dot_product))
        
        # Merge if angle is small enough
        if angle_between <= max_merge_angle:
            # Create new line from previous segment start to current segment end
            num_points = next_start - prev_start + 1
            t = np.linspace(0, 1, num_points)
            
            start_point = np.array(new_path[prev_start])
            end_point = np.array(new_path[next_start])
            
            for j in range(num_points):
                new_path[prev_start + j] = start_point + t[j] * (end_point - start_point)
            
            segment_starts.pop(i)
        else:
            i += 1
    
    # Second pass: identify and remove zigzags
    i = 1
    while i < len(segment_starts) - 2:  # Need at least 3 segments to check for zigzag
        start1 = segment_starts[i-1]
        start2 = segment_starts[i]
        start3 = segment_starts[i+1]
        end3 = segment_starts[i+2]
        
        # Calculate overall direction (first to last point)
        first_vector = np.array(path[start2]) - path[start1]
        last_vector = np.array(path[end3]) - path[start3]
        
        # Normalize vectors
        first_dir = first_vector / np.linalg.norm(first_vector)
        last_dir = last_vector / np.linalg.norm(last_vector)
        
        # Calculate angle between overall direction and initial direction
        dot_product = np.clip(np.dot(last_dir, first_dir), -1.0, 1.0)
        direction_change = np.degrees(np.arccos(dot_product))
        
        # Check if directions are similar (not opposite)
        directions_similar = dot_product > 0
        
        # Remove zigzag only if direction change is small and directions are similar
        if direction_change <= max_zigzag_deviation and directions_similar:
            num_points = end3 - start1 + 1
            t = np.linspace(0, 1, num_points)
            
            start_point = np.array(new_path[start1])
            end_point = np.array(new_path[end3])
            
            for j in range(num_points):
                new_path[start1 + j] = start_point + t[j] * (end_point - start_point)
            
            segment_starts.pop(i+1)
            segment_starts.pop(i)
        else:
            i += 1
    
    return segment_starts, new_path

def calcOrientations(middle_path, rear_path, connection_indices):
    theta_seq = []

    for i in range(1, len(middle_path)):
        moddle_point_vector = np.array([middle_path[i][0]-middle_path[i-1][0], 
                                        middle_path[i][1]-middle_path[i-1][1]])
        
        middle_rear_vector = np.array([rear_path[i][0]-middle_path[i-1][0], 
                                       rear_path[i][1]-middle_path[i-1][1]])

        angle = np.arctan2(moddle_point_vector[1], moddle_point_vector[0])

        if np.dot(moddle_point_vector, middle_rear_vector) < 0:
            angle = normalizeAngle(angle - np.pi)

        theta_seq.append(angle)
        
    theta_seq.append(theta_seq[-1])

    # # Afterprocess theta_seq
    # connection_indices = []

    # for i in range(1, len(theta_seq)):
    #     if abs(theta_seq[i] - theta_seq[i-1]) > 0.01:
    #         connection_indices.append(i)

    for idx in connection_indices:
        prev_angle = theta_seq[idx-1]  # Angle of previous segment
        curr_angle = theta_seq[idx]    # Angle of current segment
        
        # Calculate the angle that forms equal angles with both segments
        # This is the average angle between the two segments
        tangent_angle = (prev_angle + curr_angle) / 2

        # If the difference between angles is more than π radians,
        # we need to handle the wrap-around case
        angle_diff = curr_angle - prev_angle
        if abs(angle_diff) > np.pi:
            if angle_diff > 0:
                tangent_angle += np.pi
            else:
                tangent_angle -= np.pi

        theta_seq[idx] = tangent_angle

    return theta_seq


# Optimizer to fit 3 points into the robot's confiduration
class RobotConfigurationFitter:
    def __init__(self, l_vss, l_conn):
        """
        Initialize the configuration fitter.
        
        Args:
            l_vss: Length of VSS segment
            l_conn: Length of connection segment
        """
        self.l_vss = l_vss
        self.l_conn = l_conn
        
    def _compute_vss_end_pos(self, theta, k, is_front=True):
        """
        Compute the end position of VSS segment.
        
        Args:
            theta: Base orientation
            k: Curvature
            is_front: True for front segment, False for rear segment
        """
        if abs(k) < 1e-6:  # Practically straight line
            direction = -1 if is_front else 1
            return self.l_vss * direction * np.array([np.cos(theta), np.sin(theta)])
        else:
            theta_end = theta + (-1 if is_front else 1) * self.l_vss * k
            return 1/k * np.array([
                np.sin(theta_end) - np.sin(theta),
                -np.cos(theta_end) + np.cos(theta)
            ])
    
    def _compute_segment_end(self, base_pos, theta, k, is_front=True):
        """
        Compute the end position of a complete segment (VSS + connection).
        
        Args:
            base_pos: Base position [x, y]
            theta: Base orientation
            k: Curvature
            is_front: True for front segment, False for rear segment
        """
        vss_end = self._compute_vss_end_pos(theta, k, is_front)
        theta_end = theta + (-1 if is_front else 1) * self.l_vss * k
        
        direction = -1 if is_front else 1
        conn_vector = direction * self.l_conn * np.array([
            np.cos(theta_end),
            np.sin(theta_end)
        ])
        
        return base_pos + vss_end + conn_vector
    
    def _objective_function(self, q, p_0, p_1, p_2, gamma, prev_q=None, smoothness_weight=0.5, bounds=None):
        """
        Compute the objective function value.
        
        Args:
            q: Robot configuration [x, y, theta, k1, k2]
            p_0, p_1, p_2: Target positions for middle, front, and rear points
            gamma: Target orientation at middle point
            prev_q: Previous configuration for smoothness constraint
            smoothness_weight: Weight for smoothness term
            bounds: Optimization bounds for normalization
        """
        x, y, theta, k1, k2 = q
        base_pos = np.array([x, y])
        
        # Compute actual positions
        front_pos = self._compute_segment_end(base_pos, theta, k1, True)
        rear_pos = self._compute_segment_end(base_pos, theta, k2, False)
        
        # Compute objective terms
        pos_error = np.linalg.norm(base_pos - p_0)
        front_error = np.linalg.norm(front_pos - p_1)
        rear_error = np.linalg.norm(rear_pos - p_2)
        orientation_error = abs(gamma - theta)
        
        # Base objective
        objective = 10 * pos_error + front_error + rear_error + 0.001 * orientation_error
        
        # Add smoothness term if previous configuration exists
        if prev_q is not None and bounds is not None:
            # Calculate range for each variable
            ranges = np.array([bound[1] - bound[0] for bound in bounds])
            
            # Normalize the differences by their respective ranges
            normalized_diff = (q - prev_q) / ranges
            smoothness_error = np.sum(normalized_diff**2)

            objective += 0.003 * (np.abs(q[3]) + np.abs(q[4]))
            objective += smoothness_weight * smoothness_error
            
        return objective
    
    def fit_configurations(self, middle_path_points, front_path_points, 
                         rear_path_points, theta_seq, smoothness_weight=0.01):
        """
        Fit robot configurations to the given path points.
        
        Args:
            middle_path_points: List of middle point positions
            front_path_points: List of front point positions
            rear_path_points: List of rear point positions
            theta_seq: List of orientations at middle points
            smoothness_weight: Weight for smoothness constraint (default: 1.0)
            
        Returns:
            List of robot configurations [x, y, theta, k1, k2]
        """
        configurations = []
        prev_q = None
        
        for i in range(len(middle_path_points)):
            p_0 = np.array(middle_path_points[i])
            p_1 = np.array(front_path_points[i])
            p_2 = np.array(rear_path_points[i])
            gamma = theta_seq[i]
            
            # Initial guess
            if prev_q is None:
                q0 = [
                    p_0[0],  # x
                    p_0[1],  # y
                    gamma,   # theta
                    0.0,     # k1
                    0.0      # k2
                ]
            else:
                # Use previous configuration as initial guess
                q0 = prev_q
            
            # Bounds for optimization
            bounds = [
                (p_0[0] - 0.03, p_0[0] + 0.03),  # x
                (p_0[1] - 0.03, p_0[1] + 0.03),  # y
                (gamma - np.pi/2, gamma + np.pi/2),  # theta
                (-np.pi/(2*self.l_vss), np.pi/(2*self.l_vss)),  # k1
                (-np.pi/(2*self.l_vss), np.pi/(2*self.l_vss))   # k2
            ]
            
            # Optimize
            result = minimize(
                self._objective_function,
                q0,
                args=(p_0, p_1, p_2, gamma, prev_q, smoothness_weight, bounds),
                bounds=bounds,
                method='SLSQP'
            )
            
            configurations.append(result.x)
            prev_q = result.x  # Update previous configuration
            
        return configurations


# Generate feasible trajectory
def findFlatRegions(k, min_length=5, slope_threshold=0.5):
    """
    Find regions where curvature remains relatively constant.
    
    Args:
        k: curvature array
        min_length: minimum length of flat region to consider
        slope_threshold: maximum allowed slope to consider region as flat
    
    Returns:
        list of tuples (start_idx, end_idx) for each flat region
    """
    # Calculate local slopes
    slopes = np.abs(np.gradient(k))
    
    # Find regions where slope is below threshold
    flat_mask = slopes < slope_threshold
    
    # Find start and end points of flat regions
    flat_regions = []
    start_idx = None
    
    for i in range(len(flat_mask)):
        if flat_mask[i] and start_idx is None:
            start_idx = i
        elif not flat_mask[i] and start_idx is not None:
            if i - start_idx >= min_length:
                flat_regions.extend([start_idx, i-1])
            start_idx = None
    
    # Check last region
    if start_idx is not None and len(k) - start_idx >= min_length:
        flat_regions.extend([start_idx, len(k)-1])
    
    return flat_regions

def removeCloseIndices(indices, min_distance=3):
    """
    Remove indices that are too close to each other, keeping the latter one.
    
    Args:
        indices: List of indices in ascending order
        min_distance: Minimum allowed distance between indices
    
    Returns:
        List of indices with close ones removed
    """
    if len(indices) <= 1:
        return indices
    
    filtered_indices = []
    i = len(indices) - 1  # Start from the end
    
    while i >= 0:
        current = indices[i]
        # Add current index
        if not filtered_indices or current + min_distance <= filtered_indices[0]:
            filtered_indices.insert(0, current)
        i -= 1
    
    return filtered_indices

def getCrucialConfigs(q_ref_path):
    q_ref_array = np.array(q_ref_path)

    k1 = q_ref_array[:, 3]  # First segment curvature
    k2 = q_ref_array[:, 4]  # Second segment curvature

    configs_idx = [0]

    flat_regions_k1 = findFlatRegions(k1)
    flat_regions_k2 = findFlatRegions(k2)

    configs_idx.extend(flat_regions_k1)
    configs_idx.extend(flat_regions_k2)

    peak_heaight = 5

    peaks_k1, _ = find_peaks(k1, height=peak_heaight)
    valleys_k1, _ = find_peaks(-k1, height=peak_heaight)
    
    # Find local extrema of k2
    peaks_k2, _ = find_peaks(k2, height=peak_heaight)
    valleys_k2, _ = find_peaks(-k2, height=peak_heaight)
    
    # Combine all crucial points
    configs_idx.extend(peaks_k1)
    configs_idx.extend(valleys_k1)
    configs_idx.extend(peaks_k2)
    configs_idx.extend(valleys_k2)
    
    # Add last configuration
    configs_idx.append(len(q_ref) - 1)
    
    # Sort and remove duplicates
    configs_idx = sorted(list(set(configs_idx)))

    crucial_configs_idx = removeCloseIndices(configs_idx)

    # print(crucial_configs_idx)
    # print()

    # fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    # axs = axs.flatten()  # Flatten the 2D array of axes for easy indexing

    # for i in range(2):
    #     axs[i].plot(np.array(q_ref)[:, i+3], 'k-')
    #     axs[i].set_title(f'Plot of k_{i+1}')
    #     axs[i].set_xlabel('Time Step')
    #     axs[i].set_ylabel('Value')

    # # axs[0].plot(peaks_k1, q_ref[peaks_k1, 3], 'ro')
    # # axs[0].plot(valleys_k1, q_ref[valleys_k1, 3], 'bo')
    
    # # axs[1].plot(peaks_k2, q_ref[peaks_k2, 4], 'ro')
    # # axs[1].plot(valleys_k2, q_ref[valleys_k2, 4], 'bo')

    # axs[0].plot(crucial_configs_idx, q_ref[crucial_configs_idx, 3], 'ro')
    # axs[1].plot(crucial_configs_idx, q_ref[crucial_configs_idx, 4], 'ro')


    # plt.tight_layout()
    # plt.show()

    return crucial_configs_idx

# ------------------------------ Grasp Control ------------------------------

def grasp(target_pose: list) -> None:
    global agent, tracking_data, elapsed_time

    target_config = target_pose + [15, 15]
    v_prev = [0.0] * 2
    s = [1, 1]
    finish = False

    while True:
        if close2Shape(agent.curvature, target_config[3:]) or rgb_camera.finish:
            v_soft = [0.0] * 2
            s = [0, 0]
            finish = True
        else:
            v_soft = agent_controller.mpcSM3(agent, target_config, v_prev)
        
        v = [0.0] * 3 + v_soft
        _, _, s_current, _ = agent_controller.move(agent, v, s)
        agent.stiffness = s_current
        
        v_prev = v_soft

        current_time = time.perf_counter()
        elapsed_time = current_time - start_time

        tracking_data.append({'time': elapsed_time,
                              'config': agent.config.tolist(),
                              'stiffness': agent.stiffness,
                              'target_vel': v})
            
        if finish:
            break

# --------------------------- Plotting & Animation --------------------------

def plotDotPaths():
    rear_seg_points = [rear_path_points[i] for i in rear_seg_idx]
    front_seg_points = [front_path_points[i] for i in front_seg_idx]
    middle_seg_points = [middle_path_points[i] for i in middle_seg_idx]

    # Create subplots
    fig, axs = plt.subplots(1, 3, figsize=(16, 8))

    # Plot rear_path_points
    axs[0].plot(*zip(*rear_path_points), 'k--', label='Rear Path Points original')
    axs[0].plot(*zip(*rear_path_adjusted), 'r-', label='Rear Path Points smooth')
    axs[0].plot(*zip(*rear_seg_points), 'r.', label='Rear Path Points endpoints')
    # axs[0].plot(rear_smooth[:,0], rear_smooth[:,1], 'r-', label='Rear Path Points smooth')
    axs[0].set_title('Rear Path Points')
    axs[0].set_xlabel('X')
    axs[0].set_ylabel('Y')
    axs[0].axis('equal')
    axs[0].grid()
    axs[0].legend() 

    # Plot front_path_points
    axs[1].plot(*zip(*front_path_points), 'k--', label='Front Path Points original')
    axs[1].plot(*zip(*front_path_adjusted), 'g-', label='Rear Path Points smooth')
    axs[1].plot(*zip(*front_seg_points), 'g.', label='Front Path Points endpoints')
    # axs[1].plot(front_smooth[:,0], front_smooth[:,1], 'g-', label='Front Path Points smooth')
    axs[1].set_title('Front Path Points')
    axs[1].set_xlabel('X')
    axs[1].set_ylabel('Y')
    axs[1].axis('equal')
    axs[1].grid()
    axs[1].legend()

    # Plot middle_path_points
    axs[2].plot(*zip(*middle_path_points), 'k--', label='Middle Path Points original')
    axs[2].plot(*zip(*middle_path_adjusted), 'b-', label='Rear Path Points smooth')
    axs[2].plot(*zip(*middle_seg_points), 'b.', label='Middle Path Points endpoints')
    # axs[2].plot(middle_smooth[:,0], middle_smooth[:,1], 'b-', label='Middle Path Points smooth')
    axs[2].set_title('Middle Path Points')
    axs[2].set_xlabel('X')
    axs[2].set_ylabel('Y')
    axs[2].axis('equal')
    axs[2].grid()
    axs[2].legend()

    plt.tight_layout()
    plt.show()

def arc(config: list, seg=1) -> tuple[np.ndarray, np.ndarray, float]:
        k = config[2+seg]
        l = np.linspace(0, gv.L_VSS, 50)
        flag = -1 if seg == 1 else 1
        theta_array = config[2] + flag * k * l

        if abs(k) < 1e-6:
            x = np.array([0, flag * gv.L_VSS * np.cos(config[2])])
            y = np.array([0, flag * gv.L_VSS * np.sin(config[2])])
        else:
            x = np.sin(theta_array) / k - np.sin(config[2]) / k
            y = -np.cos(theta_array) / k + np.cos(config[2]) / k

        x += config[0]
        y += config[1]
        theta_end = normalizeAngle(theta_array[-1])
            
        return x, y, theta_end

def plotPath(traversed_path: list, rear_pos: list, traversed_line, plot_components) -> None:
    traversed_path[0].append(rear_pos[0])
    traversed_path[1].append(rear_pos[1])

    traversed_line.set_data(traversed_path[0], traversed_path[1])
    plot_components.append(traversed_line)
            
def plotPoints(points: list, components: tuple, plot_components) -> None:
    for point, component in zip(points, components):
        component.set_data([point[0]], [point[1]])
        plot_components.append(component)
    
def plotOrientation(middle_pos: list, orientation: float, orientation_line, plot_components) -> None:
    l = 0.05

    orientation_line.set_data([middle_pos[0], middle_pos[0] + l * np.cos(orientation)], 
                              [middle_pos[1], middle_pos[1] + l * np.sin(orientation)])
    plot_components.append(orientation_line)

def getConnEnds(x: float, y: float, theta_end: float, seg=1) -> tuple[np.ndarray, np.ndarray]:
    sign = -1 if seg == 1 else 1

    conn_start = np.array([x, y])
    conn_vec = gv.L_CONN * np.array([sign * np.cos(theta_end), sign * np.sin(theta_end)])
    conn_end = conn_start + conn_vec

    return conn_start, conn_end

def getLUCenter(corner: np.ndarray, lu_theta: float, seg=1) -> np.ndarray:
    sign = -1 if seg == 1 else 1

    lu_center = corner + gv.LU_SIDE/2 * np.array([
        sign * np.cos(lu_theta) + np.sin(lu_theta),  # x shift
        sign * np.sin(lu_theta) - np.cos(lu_theta)   # y shift
        ])
    
    return lu_center

def getLUCorners(corner: np.ndarray, lu_theta: float, seg=1) -> np.ndarray:
    lu_center = getLUCenter(corner, lu_theta, seg)

    lu_corners = lu_center + gv.LU_SIDE/2 * np.array([
        [-np.cos(lu_theta) - np.sin(lu_theta), -np.sin(lu_theta) + np.cos(lu_theta)],
        [-np.cos(lu_theta) + np.sin(lu_theta), -np.sin(lu_theta) - np.cos(lu_theta)],
        [np.cos(lu_theta) + np.sin(lu_theta), np.sin(lu_theta) - np.cos(lu_theta)],
        [np.cos(lu_theta) - np.sin(lu_theta), np.sin(lu_theta) + np.cos(lu_theta)],
        [-np.cos(lu_theta) - np.sin(lu_theta), -np.sin(lu_theta) + np.cos(lu_theta)]  # Close the square
        ])
    
    return lu_corners

def plotRobot(q: list, plot_components: list, components: tuple) -> None:
    vss1_line, vss2_line, conn1_line, conn2_line, lu1_square, lu2_square = components
    
    # Plot VSS
    x_vss1, y_vss1, theta_vss1_end = arc(q, seg=1)
    vss1_line.set_data(x_vss1, y_vss1)
    plot_components.append(vss1_line)

    x_vss2, y_vss2, theta_vss2_end = arc(q, seg=2)
    vss2_line.set_data(x_vss2, y_vss2)
    plot_components.append(vss2_line)

    # Plot connection lines
    # Front connection
    conn1_start, conn1_end = getConnEnds(x_vss1[-1], y_vss1[-1], theta_vss1_end)
    conn1_line.set_data([conn1_start[0], conn1_end[0]], [conn1_start[1], conn1_end[1]])
    plot_components.append(conn1_line)
    
    # Rear connection
    conn2_start, conn2_end = getConnEnds(x_vss2[-1], y_vss2[-1], theta_vss2_end, 2)
    conn2_line.set_data([conn2_start[0], conn2_end[0]], [conn2_start[1], conn2_end[1]])
    plot_components.append(conn2_line)

    # Plot LU squares
    # Front LU
    lu1_corners = getLUCorners(conn1_end, theta_vss1_end)
    lu1_square.set_data(lu1_corners[:, 0], lu1_corners[:, 1])
    plot_components.append(lu1_square)

    # Rear LU - connected at left top corner
    lu2_corners = getLUCorners(conn2_end, theta_vss2_end, 2)
    lu2_square.set_data(lu2_corners[:, 0], lu2_corners[:, 1])
    plot_components.append(lu2_square)

def runAnimation(rear_path_points, front_path_points, middle_path_points, theta_seq, q_ref, q_optim=None, s_optimal=None) -> None:
    # Colors
    red_color = '#ec5353'
    blue_color = '#3471A8'
    grey_color = '#474747'

    # Rear point with its traversed path
    traversed_line, = plt.plot([], [], '-', color=red_color, linewidth=3)
    rear_point, = plt.plot([], [], 'o', color=red_color, markersize=6)

    # Front and niddle points with orientation at the middle point
    front_point, = plt.plot([], [], 'o', color=blue_color, markersize=6)
    middle_point, = plt.plot([], [], 'o', color=blue_color, markersize=7)
    orientation_line, = plt.plot([], [], color=blue_color, linewidth=2)

    # Robot components
    vss1_line, = plt.plot([], [], color=grey_color, linewidth=2)
    vss2_line, = plt.plot([], [], color=grey_color, linewidth=2)
    conn1_line, = plt.plot([], [], color=grey_color, linewidth=2)
    conn2_line, = plt.plot([], [], color=grey_color, linewidth=2)
    lu1_square, = plt.plot([], [], color=grey_color, linewidth=2)
    lu2_square, = plt.plot([], [], color=grey_color, linewidth=2)

    vss1_line_optim, = plt.plot([], [], color=grey_color, linewidth=2)
    vss2_line_optim, = plt.plot([], [], color=grey_color, linewidth=2)
    conn1_line_optim, = plt.plot([], [], color=grey_color, linewidth=2)
    conn2_line_optim, = plt.plot([], [], color=grey_color, linewidth=2)
    lu1_square_optim, = plt.plot([], [], color=grey_color, linewidth=2)
    lu2_square_optim, = plt.plot([], [], color=grey_color, linewidth=2)

    class AnimationState:
        def __init__(self):
            self.iteration = 0
            self.idx = 0
            self.traversed_path = [[], []]
            self.counter = 0
            self.pause_frames_n = 5
    
    state = AnimationState()

    def animate(frame):
        plot_components = []
        
        if state.idx < len(rear_path_points):
            rear_pos = rear_path_points[state.idx]
            front_pos = front_path_points[state.idx]
            middle_pos = middle_path_points[state.idx]
            orientation = theta_seq[state.idx]

            q_ref_i = q_ref[state.idx]
            if q_optim is not None and s_optimal is not None:
                q_optim_i = q_optim[state.idx]
                s_optim_i = s_optimal[state.idx]
            
        if state.iteration == 0:
            # Iteration 0: Just show Voronoi diagram (empty plot_components)
            if state.counter > state.pause_frames_n:
                state.counter = 0
                state.iteration += 1
            else:
                state.counter += 1
        
        elif state.iteration == 1:
            # Iteration 1: rear point and its path only
            plotPath(state.traversed_path, rear_pos, traversed_line, plot_components)
            rear_point.set_color(red_color)
            plotPoints([rear_pos], (rear_point,), plot_components)

            if state.idx < len(rear_path_points)-1:
                state.idx += 1
            else:
                if state.counter > state.pause_frames_n:
                    state.counter = 0
                    state.idx = 0
                    state.iteration += 1

                    state.traversed_path = [[], []]
                else:
                    state.counter += 1
            
        elif state.iteration == 2:
            # Iteration 2: all points, path, and orientation
            plotPath(state.traversed_path, rear_pos, traversed_line, plot_components)
            plotPoints([front_pos, middle_pos, rear_pos], 
                        (front_point, middle_point, rear_point), plot_components)
            plotOrientation(middle_pos, orientation, orientation_line, plot_components)

            if state.idx < len(rear_path_points)-1:
                state.idx += 1
            else:
                if state.counter > state.pause_frames_n:
                    state.counter = 0
                    state.idx = 0
                    state.iteration += 1

                    state.traversed_path = [[], []]
                    traversed_line.set_data(state.traversed_path)
                else:
                    state.counter += 1
            
        elif state.iteration == 3:
            # Iteration 3: all points, orientation, robot, rear point in blue
            rear_point.set_color(blue_color)
            plotPoints([front_pos, middle_pos, rear_pos], 
                        (front_point, middle_point, rear_point), plot_components)
            plotOrientation(middle_pos, orientation, orientation_line, plot_components)
            
            robot_ref_components = (vss1_line, vss2_line, conn1_line, conn2_line, 
                                    lu1_square, lu2_square)
            plotRobot(q_ref_i, plot_components, robot_ref_components)

            if state.idx < len(rear_path_points)-1:
                state.idx += 1
            else:
                if state.counter > state.pause_frames_n:
                    state.counter = 0
                    state.idx = 0
                    state.iteration += 1

                    front_point.set_data([[], []])
                    middle_point.set_data([[], []])
                    rear_point.set_data([[], []])
                    orientation_line.set_data([[], []])
                else:
                    state.counter += 1
        
        elif q_optim is not None and s_optimal is not None:
            robot_ref_components = (vss1_line, vss2_line, conn1_line, conn2_line, 
                                    lu1_square, lu2_square)
            for robot_ref_component in robot_ref_components:
                robot_ref_component.set_alpha(0.5)
            plotRobot(q_ref_i, plot_components, robot_ref_components)

            robot_optim_components = (vss1_line_optim, vss2_line_optim, conn1_line_optim, conn2_line_optim, 
                                      lu1_square_optim, lu2_square_optim)
            
            if s_optim_i[0] == 1:
                vss1_line_optim.set_color(red_color)
            else:
                vss1_line_optim.set_color(grey_color)
            
            if s_optim_i[1] == 1:
                vss2_line_optim.set_color(red_color)
            else:
                vss2_line_optim.set_color(grey_color)

            plotRobot(q_optim_i, plot_components, robot_optim_components)

            if state.idx < len(rear_path_points)-1:
                state.idx += 1


        return tuple(plot_components)

    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, frames=4*len(rear_path_points)+30,
        interval=1,  # 50ms between frames
        blit=True,
        repeat=False
    )
    anim.save('D:/Robot 2SR/2sr-swarm-control/Experiments/Video/Grasping/traverse_animation.mp4', writer='ffmpeg', fps=8)  # Save the animation as an MP4 file
    
    plt.title('Voronoi Diagram with 3-Point Segment Motion')
    plt.show()

# ---------------------------------------------------------------------------



if __name__ == "__main__":

    # -------------------------------- Load Data --------------------------------
    
    if os.path.exists('voronoi_plot.pkl') and os.path.getsize('voronoi_plot.pkl') > 0:
        with open('voronoi_plot.pkl', 'rb') as file:
            plot_data = pickle.load(file)
            fig = plot_data['figure']
            ax = plot_data['axes']
        
    # Define the file path to
    file_path = 'path_points.json'

    # Function to save path points to a file
    def savePathPoints(rear, front, middle):
        with open(file_path, 'w') as f:
            json.dump({'rear': rear, 'front': front, 'middle': middle}, f)

    # Function to load path points from a file
    def load_path_points():
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            with open(file_path, 'r') as f:
                return json.load(f)
        return None

    # Load existing path points if available
    loaded_path_points = load_path_points()
    if loaded_path_points:
        rear_path_points = loaded_path_points['rear']
        front_path_points = loaded_path_points['front']
        middle_path_points = loaded_path_points['middle']
    else:
        rear_path_points, front_path_points, middle_path_points = setUpEnv()
        
    # -------------------------------- Process Data --------------------------------
    
    # Adjust each path
    rear_seg_idx, rear_path_adjusted = adjustPath(rear_path_points)
    front_seg_idx, front_path_adjusted = adjustPath(front_path_points)
    middle_seg_idx, middle_path_adjusted = adjustPath(middle_path_points)

    # plotDotPaths()

    theta_seq = calcOrientations(middle_path_adjusted, rear_path_adjusted, middle_seg_idx[1:-1])

    # Fit robot's configuration
    print('Calculate the robot\'s reference path...')
    print()

    fitter = RobotConfigurationFitter(gv.L_VSS, gv.L_CONN + gv.LU_SIDE)
    q_ref = fitter.fit_configurations(middle_path_adjusted, front_path_adjusted,
                                      rear_path_adjusted, theta_seq)

    print('Determine crucial shapes...')
    print()
    crucial_configs_idx = getCrucialConfigs(q_ref)

    crucial_configs = [q_ref[i] for i in crucial_configs_idx]
    # print(len(crucial_configs))
    # print()
    
    s_list = []
    for i in range(len(crucial_configs)-1):
        s = [0, 0]

        if abs(crucial_configs[i][3] - crucial_configs[i+1][3]) > 3:
            s[0] = 1
        if abs(crucial_configs[i][4] - crucial_configs[i+1][4]) > 3:
            s[1] = 1

        s_list.append(s)
    s_list.append([0, 0])

    print(s_list)
    print()

    # crucial_configs_anim = []

    # for i in range(len(crucial_configs_idx)-1):
    #     crucial_configs_anim.extend([q_ref[crucial_configs_idx[i]]] * (crucial_configs_idx[i+1] - crucial_configs_idx[i]))
    # crucial_configs_anim.append(q_ref[crucial_configs_idx[-1]])

    # s_optimal = [[0, 0]] * len(crucial_configs_anim)
    
    # print('Start animation...')
    # print()

    # if rear_path_points:
    #     runAnimation(rear_path_adjusted, front_path_adjusted, middle_path_adjusted, theta_seq, 
    #                  q_ref, crucial_configs_anim, s_optimal)

    # print()
    # print('Finished!')


    # kp = [50, 50, 1, 0.5, 0.5]

    # initial_config = crucial_configs[1]
    # target_config = crucial_configs[2]

    # num_intermediate = 2
    # t = np.linspace(0, 1, num_intermediate + 2)

    # # Special handling for angle (theta) to ensure proper interpolation
    # # Unwrap theta to prevent interpolation issues around ±π
    # theta_initial = initial_config[2]
    # theta_target = target_config[2]
    # delta_theta = (theta_target - theta_initial + np.pi) % (2 * np.pi) - np.pi
    # theta_interp = theta_initial + delta_theta * t[1:-1]
    
    # # Interpolate all configurations
    # target_configs = []
    
    # for i in range(num_intermediate):
    #     new_config = initial_config + (target_config - initial_config) * t[i+1]
    #     # Replace theta with properly interpolated angle
    #     new_config[2] = theta_interp[i]
    #     new_config = np.round(new_config, 4)
    #     target_configs.append(new_config.tolist())
    
    # target_configs.append(np.round(target_config, 4).tolist())

    # print(target_configs)

    # current_config = np.array(initial_config)
    # agent_sim = rsr.Robot(2, *initial_config)

    
    # q_tilda = np.array(target_config) - current_config
    # diff = [np.linalg.norm(q_tilda)]

    # max_iter = 50
    # counter = 0

    # while counter < max_iter:
    #     J = agent_sim.jacobian([1, 1])
    #     v = np.linalg.pinv(J) @ (kp * q_tilda)

    #     q_dot = J @ v * gv.DT
    #     current_config += q_dot
    #     agent_sim.config = current_config

    #     q_tilda = np.array(target_config) - current_config
    #     diff.append(np.linalg.norm(q_tilda))
    #     counter += 1




    # fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    # grey_color = '#474747'
    # plot_components = []

    # vss1_line, = ax[0].plot([], [], color=grey_color, linewidth=2)
    # vss2_line, = ax[0].plot([], [], color=grey_color, linewidth=2)
    # conn1_line, = ax[0].plot([], [], color=grey_color, linewidth=2)
    # conn2_line, = ax[0].plot([], [], color=grey_color, linewidth=2)
    # lu1_square, = ax[0].plot([], [], color=grey_color, linewidth=2)
    # lu2_square, = ax[0].plot([], [], color=grey_color, linewidth=2)

    # robot_ref_components = (vss1_line, vss2_line, conn1_line, conn2_line, lu1_square, lu2_square)

    # vss1_line_target, = ax[0].plot([], [], color=grey_color, linewidth=2, alpha=0.5)
    # vss2_line_target, = ax[0].plot([], [], color=grey_color, linewidth=2, alpha=0.5)
    # conn1_line_target, = ax[0].plot([], [], color=grey_color, linewidth=2, alpha=0.5)
    # conn2_line_target, = ax[0].plot([], [], color=grey_color, linewidth=2, alpha=0.5)
    # lu1_square_target, = ax[0].plot([], [], color=grey_color, linewidth=2, alpha=0.5)
    # lu2_square_target, = ax[0].plot([], [], color=grey_color, linewidth=2, alpha=0.5)

    # robot_target_components = (vss1_line_target, vss2_line_target, conn1_line_target, 
    #                            conn2_line_target, lu1_square_target, lu2_square_target)


    # plotRobot(current_config, plot_components, robot_ref_components)
    # plotRobot(target_config, plot_components, robot_target_components)

    # ax[1].plot(diff, 'k-', linewidth=1)

    # ax[0].axis('equal')
    # ax[0].set_xlim(-0.7, 0.3)  # Set x-axis range
    # ax[0].set_ylim(-0.7, 0.3)  # Set y-axis range

    # ax[1].axis('equal')
    # # ax[1].set_xlim(-1, 15)  # Set x-axis range
    # # ax[1].set_ylim(0, len(diff))  # Set y-axis range

    # plt.tight_layout()
    # plt.show()

    setUpEnv()

    # v_r = [0.0] * 5
    # s = [0, 0]

    # for target_config in crucial_configs[:2]:
    #     rgb_camera.target_robot_config = target_config
    #     finish = False

    #     while True:
    #         if close2Goal(agent.pose, target_config[:3]) or rgb_camera.finish:
    #             v_r = [0.0] * 3
    #             q_new = agent.pose
    #             finish = True
    #         else:
    #             v_r, q_new = agent_controller.mpcRM(agent, target_config[:3], v_r)

    #         if simulation:
    #             agent.pose = q_new

    #         v = v_r + [0.0, 0.0]
    #         # print(v_r)
    #         agent_controller.move(agent, v, s)

    #         if finish:
    #             break
        
    #     if rgb_camera.finish:
    #         break

    #     print('Next target!')
    #     print()


    initial_config = agent.config
    target_config = crucial_configs[4]

    num_intermediate = 2
    t = np.linspace(0, 1, num_intermediate + 2)

    # Special handling for angle (theta) to ensure proper interpolation
    # Unwrap theta to prevent interpolation issues around ±π
    theta_initial = initial_config[2]
    theta_target = target_config[2]
    delta_theta = (theta_target - theta_initial + np.pi) % (2 * np.pi) - np.pi
    theta_interp = theta_initial + delta_theta * t[1:-1]
    
    # Interpolate all configurations
    target_configs = []
    
    for i in range(num_intermediate):
        new_config = initial_config + (target_config - initial_config) * t[i+1]
        # Replace theta with properly interpolated angle
        new_config[2] = theta_interp[i]
        new_config = np.round(new_config, 4)
        target_configs.append(new_config.tolist())

    target_configs.append(np.round(target_config, 4).tolist())

    print(target_configs)

    target_configs = [grasp_pose + [15, 15], 
                      [grasp_pose[0], grasp_pose[1] + 0.3, grasp_pose[2], 15, 15]]

    
    for tc in target_configs:
        rgb_camera.target_robot_config = tc
        finish = False

        v_r = [0.0] * 3
        v_s = [0.0] * 2

        s = [0, 0]
        for i in range(2):
            if abs(tc[i+3] - agent.curvature[i]) > 5:
                s[i] = 1

        while True:
            if s != [0, 0] and close2Shape(agent.curvature, tc[3:]):
                s = [0, 0]
                v_r = [0.0] * 3
                v_s = [0.0] * 2
            if (s == [0, 0] and close2Goal(agent.pose, tc[:3])) or rgb_camera.finish:
                v_r = [0.0] * 3
                v_s = [0.0] * 2
                finish = True
            else:
                if s == [0, 0]:
                    v_r, q_new = agent_controller.mpcRM(agent, tc[:3], v_r)
                elif s == [1, 0]:
                    v_s, q_new = agent_controller.mpcSM1(agent, tc, v_s)
                elif s == [0, 1]:
                    v_s, q_new = agent_controller.mpcSM2(agent, tc, v_s)
                else:
                    v_s, q_new = agent_controller.mpcSM3(agent, tc, v_s)

            if simulation:
                agent.config = q_new

            v = v_r + v_s
            print(f'Stiffness: {s}')
            print(f'Velocity: {v}')

            agent_controller.move(agent, v, s)
            print()

            if finish:
                break
        
        if rgb_camera.finish:
            break

        print('Next target!')
        print()
    

    print('Finish!')


        

    

    

    
    
    



    
