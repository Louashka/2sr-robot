import sys
sys.path.append('D:/Robot 2SR/2sr-swarm-control')
from Model import global_var as gv, robot2sr as rsr, manipulandum, splines
import motive_client, robot2sr_controller as rsr_ctrl, camera_optitrack_synchronizer as cos
import threading
from datetime import datetime
import time
from shapely.geometry import Polygon, Point, LineString
from shapely import affinity
import cv2
import numpy as np
import cvxpy
import json
from typing import List
from scipy.optimize import minimize
from skopt import gp_minimize
from scipy.spatial import Voronoi
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as PlotPolygon
from matplotlib.collections import PatchCollection
from matplotlib import animation
from collections import deque

agent: rsr.Robot = None
object: manipulandum.Shape = None

mocap = motive_client.MocapReader()
rgb_camera = cos.Aligner()
agent_controller = rsr_ctrl.Controller()

markers = None
agent_width = 0.07
agent_length = 0.34

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

simulation = True
# simulation = False

def updateConfig():
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

def updateConfigLoop():

    while True:
        updateConfig()

        rgb_camera.markers = markers
        if object is not None:
            rgb_camera.manip_center = object.pose

def expandObstacles():
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

def normalizeAngle(angle):
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

def closeToGoal(current, target):
    status = True

    # Calculate Euclidean distance between current and target (x, y)
    distance = np.linalg.norm(np.array(current) - np.array(target))
    
    # Define thresholds for position and orientation
    distance_threshold = 0.01

    if distance > distance_threshold:
        status = False
    
    print(f"Distance to goal: {distance:.3f} m")
    print()

    return status

def traversePath(path):
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
        if closeToGoal(agent.position, traj.getPoint(-1)) or rgb_camera.finish:
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

def getVelAlongHeading(v, angle):
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

def getLinearModelMatrix(vref, phi):
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

def mpc(qref, vref, heading_angle):
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

def grasp(target_pose):
    global agent, tracking_data, elapsed_time

    target_config = target_pose + [15, 15]
    v_prev = [0.0] * 2
    s = [1, 1]
    finish = False

    while True:
        if closeToShape(agent.curvature, target_config[3:]) or rgb_camera.finish:
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

def closeToShape(current_k, target_k):
    status = True
    k1_diff = abs(current_k[0] - target_k[0])
    k2_diff = abs(current_k[1] - target_k[1])
    if k1_diff > 5 or k2_diff > 5:
        status = False
    return status

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
    
    def is_line_collision_free(self, p1, p2):
        """Check if line segment between p1 and p2 collides with any obstacle"""
        line = LineString([p1, p2])
        for obstacle in self.obstacles:
            if line.intersects(obstacle):
                return False
        return True

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
        
        return passages
    
    def is_point_collision_free(self, point):
        """Check if point is inside any obstacle"""
        point_obj = Point(point)
        for obstacle in self.obstacles:
            if obstacle.contains(point_obj) or obstacle.boundary.contains(point_obj):
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
        # First, calculate target positions for all three points when segment reaches goal
        mid_point_target = goal_pose[:-1]
        segment_half = segment_length / 2

        front_start = np.array(start_pose[:-1]) + segment_half * np.array([-np.cos(start_pose[-1]), -np.sin(start_pose[-1])])
        rear_start = np.array(start_pose[:-1]) + segment_half * np.array([np.cos(start_pose[-1]), np.sin(start_pose[-1])])

        # Calculate target positions for endpoints
        front_target = np.array(mid_point_target) + segment_half * np.array([-np.cos(goal_pose[-1]), -np.sin(goal_pose[-1])])
        rear_target = np.array(mid_point_target) + segment_half * np.array([np.cos(goal_pose[-1]), np.sin(goal_pose[-1])])
        
        # Find path for rear point (leading the motion)
        rear_path = self.find_point_path(rear_start, rear_target)
        front_path = self.find_point_path(front_start, front_target)

        print(rear_path)
        print(front_path)
        
        if not rear_path:
            return None
        
        interpoltaed_rear_path_points = self.interpolate_path_points(rear_path, rear_target)
        front_path_points = []

        path_between_rear_front = self.find_point_path(rear_path[0]['points'][0],
                                                       front_path[0]['points'][1])
        passages_between_rear_front = deque()
        for item in path_between_rear_front[:-1]:
            passages_between_rear_front.append(item['nodes'])

        front_current_idx = 0

        for rear_path_point in interpoltaed_rear_path_points:
            rear_pos = rear_path_point['pos']
            front_pos_new = None

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

                for i in range(len(passages_between_rear_front)):
                    passage = passages_between_rear_front[i]
                    if i == 0:
                        passage_start_pos = rear_pos
                    else:
                        passage_start_pos = self.allowed_passage_graph.nodes[passage[0]]['pos']

                    delta_rear_front += np.linalg.norm(self.allowed_passage_graph.nodes[passage[1]]['pos'] - passage_start_pos)

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
                        for neighbor in self.allowed_passage_graph.neighbors(passages_between_rear_front[-1][1]):
                            if neighbor != passages_between_rear_front[-1][0]:
                                nodes_to_append = (passages_between_rear_front[-1][1], neighbor)
                                break
                                
                    if nodes_to_append is not None:
                        passages_between_rear_front.append(nodes_to_append)


            front_path_points.append(front_pos_new)

        return interpoltaed_rear_path_points, front_path_points
    
    def interpolate_path_points(self, path, target_point, step_size=0.01, threshold=0.01):
        interpolated_points = []

        for i in range(len(path)-1):
            p1, p2 = path[i]['points']
            current_nodes = path[i]['nodes']

            passage_vector = np.array(p2) - np.array(p1)
            passage_length = np.linalg.norm(passage_vector)
            direction = passage_vector / passage_length
            
            # How many points we need on this passage
            num_points = int(passage_length / step_size)

            for j in range(num_points):
                point_pos = np.array(p1) + j * step_size * direction

                dist_to_target = np.linalg.norm(np.array(target_point) - point_pos)
                if dist_to_target < threshold:
                    return interpolated_points
                
                interpolated_points.append({
                    'pos': point_pos,
                    'nodes': current_nodes
                })

        return interpolated_points
    
    def find_nearest_passage(self, point):
        """Find nearest passage to a point"""
        # Convert voronoi vertices to array for efficient computation
        point_array = np.array(point)
        nearest_passage = None
        min_distance = float('inf')
        
        # Check each edge in the passage graph
        for edge in self.allowed_passage_graph.edges():
            v1, v2 = edge
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
    
    def calculate_passage_orientation(self, passage):
        """Calculate orientation of a passage"""
        if not passage:
            return 0
            
        # Get passage endpoints
        p1, p2 = passage['points']
        
        # Calculate orientation angle from p1 to p2
        return np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
    
    def find_point_path(self, start_point, goal_point):
        """Find path for a point through passages"""
        # Find nearest passages to start and goal
        start_passage = self.find_nearest_passage(start_point)
        goal_passage = self.find_nearest_passage(goal_point)
        
        if not start_passage or not goal_passage:
            return None
            
        # Get nodes from passages
        start_nodes = start_passage['nodes']
        goal_nodes = goal_passage['nodes']
        
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
        
        # Add initial passage from start point to first node
        first_node = shortest_path[0]
        first_pos = self.allowed_passage_graph.nodes[first_node]['pos']
        passage_sequence.append({
            'points': (start_point, first_pos),
            'nodes': (None, first_node),
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
        
        # Add final passage from last node to goal point
        last_node = shortest_path[-1]
        last_pos = self.allowed_passage_graph.nodes[last_node]['pos']
        passage_sequence.append({
            'points': (last_pos, goal_point),
            'nodes': (last_node, None),
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
                        markersize=2, label='Sample Points')
        
        # Plot finite Voronoi edges
        for simplex in self.vor.ridge_vertices:
            if -1 not in simplex:
                self.ax.plot(self.vor.vertices[simplex, 0], 
                           self.vor.vertices[simplex, 1], 
                           'b-', linewidth=1)
        
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
                           'b--', linewidth=1)

    def highlight_narrow_passages(self, passages, color='red', linewidth=2):
        """Highlight identified narrow passages"""
        for passage in passages:
            p1, p2 = passage['points']
            self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                        color=color, linewidth=linewidth, 
                        label=f'Passage (clearance: {passage["clearance"]:.2f})')
            
        # Remove duplicate labels
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax.legend(by_label.values(), by_label.keys())

if __name__ == "__main__":

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

    # tuner = BayesianTuner(workspace_bounds, agent.pose, target_pose)
    # best_params = tuner.tune_parameters()

    # print(best_params)

    # while True:
    #     # rrt = RRT(agent.pose, target_pose, workspace_bounds)
    #     rrt = RRTStar(agent.pose, target_pose, workspace_bounds)
        
    #     path = rrt.plan()

    #     if path is not None:
    #         print("Path found!")
    #         path.append(grasp_pose)
    #         # path.append(grasp_pose[:-1])
    #         print(path)
    #         rgb_camera.rrt_path = path
            
    #         # Ask for user confirmation
    #         user_input = input("Is this path acceptable? (yes/no): ").lower()
            
    #         if user_input == 'yes' or user_input == 'y':
    #             # Continue with the original code
    #             interpolated_path = []
    #             num_interpolated_points = 20

    #             for i in range(len(path) - 1):
    #                 start = path[i]
    #                 end = path[i + 1]
    #                 for j in range(num_interpolated_points + 1):
    #                     t = j / num_interpolated_points
    #                     x = start[0] + t * (end[0] - start[0])
    #                     y = start[1] + t * (end[1] - start[1])
    #                     theta = normalizeAngle(start[2] + t * (end[2] - start[2]))
    #                     interpolated_path.append([x, y, theta])

    #             # print('Traverse the path...')
    #             # traversePath(interpolated_path)
    #             # print('Arrived!')
    #             # print()

    #             # print('Grasp the object...')
    #             # grasp(path[-1])
    #             # print(f'Finished! Recording time: {elapsed_time} seconds')
    #             # rgb_camera.finish = True

    #             # # Prepare data to be written
    #             # data_json = {
    #             #     "metadata": {
    #             #         "description": "Grasp through obstacles",
    #             #         "date": date_title
    #             #     },
    #             #     'path': path_data,
    #             #     "tracking": tracking_data
    #             # } 

    #             # # Write data to JSON file
    #             # if not simulation:
    #             #     with open(filename, 'w') as f:
    #             #         json.dump(data_json, f, indent=2)

    #             #     print(f"Data written to {filename}")                   
                
    #             break 
                
    #         elif user_input == 'no' or user_input == 'n':
    #             rgb_camera.rrt_path = None
    #             print("Replanning path...")
    #             continue  # Continue to next iteration to find a new path
                
    #         else:
    #             print("Invalid input. Please enter 'yes' or 'no'.")
    #             continue  # Ask for input again if invalid
                
    #     else:
    #         print("No path found!")
    #         # break  # Exit if no path is found

    # ---------------------------------------------------------------
    # Create visualizer
    bounds = [
        (-0.606, -0.8), # Lower left corner (x_min, y_min)
        (0.319, 0.5)  # Upper right corner (x_max, y_max)
    ]

    visualizer = VoronoiVisualizer(expanded_obstacles, bounds)
    
    # Plot basic Voronoi diagram
    fig, ax = visualizer.plot_voronoi(show_points=True)
    
    # Create analyzer and find passages
    analyzer = VoronoiPassageAnalyzer(expanded_obstacles, bounds)
    passages = analyzer.identify_passages()
    
    # Highlight narrow passages
    visualizer.highlight_narrow_passages(passages)
    
    # Find passage sequence
    # passage_sequence = analyzer.find_passage_sequence(agent.position, target_pose[:-1], agent_length)
    rear_path_points, front_path_points = analyzer.find_passage_sequence(agent.pose, target_pose[:-1], agent_length)

    if rear_path_points:
        rear_point, = plt.plot([], [], 'bo', markersize=8)
        front_point, = plt.plot([], [], 'ko', markersize=6)
        traversed_line, = plt.plot([], [], 'g-', linewidth=3)

        traversed_path_x = []
        traversed_path_y = []

        def animate(frame):
            if frame < len(rear_path_points):
                rear_pos = rear_path_points[frame]['pos']
                front_pos = front_path_points[frame]

                traversed_path_x.append(rear_pos[0])
                traversed_path_y.append(rear_pos[1])

                traversed_line.set_data(traversed_path_x, traversed_path_y)
                rear_point.set_data([rear_pos[0]], [rear_pos[1]])
                front_point.set_data([front_pos[0]], [front_pos[1]])

            return traversed_line, rear_point, front_point

        # Create animation
        anim = animation.FuncAnimation(
            fig, animate, frames=101,
            interval=50,
            blit=True,
            repeat=False
        )
        anim.save('D:/Robot 2SR/2sr-swarm-control/Experiments/Video/Grasping/traverse_animation.mp4', writer='ffmpeg', fps=20)  # Save the animation as an MP4 file
        
        plt.title('Voronoi Diagram with 3-Point Segment Motion')
    else:
        plt.title('No valid path found')

    plt.show()
