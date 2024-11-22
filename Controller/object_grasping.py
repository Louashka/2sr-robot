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

agent: rsr.Robot = None
object: manipulandum.Shape = None
obstacles = []

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

# class Node:
#     def __init__(self, pose):
#         self.pose = np.array(pose)  # [x, y, theta]
#         self.parent = None
#         self.cost = 0.0

class RRTStar:
    def __init__(self, start_pose, target_pose, workspace_bounds, 
                 max_iter=1000, step_size=0.5, search_radius=2.0):
        self.start = Node(start_pose)
        self.goal = Node(target_pose)
        self.bounds = workspace_bounds  # [x_min, x_max, y_min, y_max]
        self.max_iter = max_iter
        self.step_size = step_size
        self.search_radius = search_radius
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
                sampled_pose = self.goal.pose
            else:
                sampled_pose = self.sampleRandomPose()
            
            # Find nearest node
            nearest_node = self.getNearestNode(sampled_pose)
            
            # Extend towards sampled pose
            new_pose = self.steer(nearest_node.pose, sampled_pose)
            
            # Check if new pose is valid
            if self.isValidPose(new_pose) and self.isPathValid(nearest_node.pose, new_pose):
                new_node = Node(new_pose)
                
                # Find nearby nodes for rewiring
                nearby_nodes = self.getNearbyNodes(new_pose)
                
                # Connect to best parent
                min_cost = float('inf')
                best_parent = None
                
                for node in nearby_nodes:
                    potential_cost = (node.cost + 
                                self.calculateDistance(node.pose, new_pose))
                    
                    if (potential_cost < min_cost and 
                        self.isPathValid(node.pose, new_pose)):
                        min_cost = potential_cost
                        best_parent = node
                
                if best_parent is not None:
                    new_node.parent = best_parent
                    new_node.cost = min_cost
                    self.nodes.append(new_node)
                    rgb_camera.all_nodes = self.nodes
                    
                    # Rewire nearby nodes
                    self.rewire(new_node, nearby_nodes)
                    
                    # Check if we can connect to goal
                    if (self.calculateDistance(new_pose, self.goal.pose) < self.step_size and 
                        self.isPathValid(new_pose, self.goal.pose)):
                        self.goal.parent = new_node
                        self.goal.cost = (new_node.cost + 
                                        self.calculateDistance(new_pose, self.goal.pose))
                        
                        # Add goal to nodes list temporarily for final rewiring
                        self.nodes.append(self.goal)
                        rgb_camera.all_nodes = self.nodes
                        
                        # Perform final rewiring
                        nearby_nodes = self.getNearbyNodes(self.goal.pose)
                        self.rewire(self.goal, nearby_nodes)
                        
                        # Remove goal from nodes list
                        self.nodes.pop()
                        rgb_camera.all_nodes = self.nodes
                        
                        return self.extractPath()
        
        return None  # No path found
    
    def sampleRandomPose(self):
        x = np.random.uniform(self.bounds[0], self.bounds[1])
        y = np.random.uniform(self.bounds[2], self.bounds[3])
        previous_theta = self.nodes[-1].pose[2] if self.nodes else agent.theta  # Get the last node's theta or default to 0
        theta = np.random.uniform(-np.pi, np.pi)
        theta = np.clip(theta, previous_theta - np.pi/2, previous_theta + np.pi/2)  # Limit theta
        return np.array([x, y, theta])
    
    def getNearestNode(self, pose):
        distances = [self.calculateDistance(node.pose, pose) for node in self.nodes]
        return self.nodes[np.argmin(distances)]
    
    def getNearbyNodes(self, pose):
        nearby = []
        for node in self.nodes:
            if self.calculateDistance(node.pose, pose) <= self.search_radius:
                nearby.append(node)
        return nearby
    
    def steer(self, from_pose, to_pose):
        dist = self.calculateDistance(from_pose, to_pose)
        if dist <= self.step_size:
            return to_pose
        else:
            ratio = self.step_size / dist
            dx = to_pose[0] - from_pose[0]
            dy = to_pose[1] - from_pose[1]
            dtheta = normalizeAngle(to_pose[2] - from_pose[2])
            
            new_x = from_pose[0] + dx * ratio
            new_y = from_pose[1] + dy * ratio
            new_theta = normalizeAngle(from_pose[2] + dtheta * ratio)
            
            return np.array([new_x, new_y, new_theta])
    
    def rewire(self, new_node, nearby_nodes):
        for node in nearby_nodes:
            potential_cost = (new_node.cost + 
                            self.calculateDistance(new_node.pose, node.pose))
            
            if (potential_cost < node.cost and 
                self.isPathValid(new_node.pose, node.pose)):
                node.parent = new_node
                node.cost = potential_cost
    
    def calculateDistance(self, pose1, pose2):
        # Euclidean distance for position, weighted angular difference
        pos_diff = np.sqrt((pose1[0] - pose2[0])**2 + (pose1[1] - pose2[1])**2)
        # angle_diff = abs(normalizeAngle(pose1[2] - pose2[2]))
        # return pos_diff + 0.2 * angle_diff  # Weight for angular component
        return pos_diff
    
    def isValidPose(self, pose):
        """Two-phase collision checking"""
        x, y = pose[0], pose[1]
        
        # Check bounds
        if not (self.bounds[0] <= x <= self.bounds[1] and 
                self.bounds[2] <= y <= self.bounds[3]):
            return False

        # Quick check with expanded obstacles
        point = Point(x, y)
        if any(obs.contains(point) for obs in expanded_obstacles):
            return False

        # If close to expanded obstacles, do precise check
        for exp_obs, orig_obs in zip(expanded_obstacles, original_obstacles):
            if exp_obs.distance(point) < agent_length:
                # Only create robot polygon if we're close to obstacles
                robot = self.getRobotPolygon(pose)
                if robot.intersects(orig_obs):
                    return False

        return True
    
    def isPathValid(self, pose1, pose2, check_steps=10):
        """Check if path between poses is collision-free"""
        # Calculate total distance and angle difference
        dist = np.sqrt((pose2[0] - pose1[0])**2 + (pose2[1] - pose1[1])**2)
        angle_diff = normalizeAngle(pose2[2] - pose1[2])

        print(angle_diff/dist)
        if abs(angle_diff/dist) > 3:
            return False
        
        # Determine number of checks based on distance and angle
        num_steps = max(
            check_steps,
            int(dist / (agent_width/2)),
            int(abs(angle_diff) / (np.pi/8))
        )

        for i in range(num_steps + 1):
            t = i / num_steps
            # Linear interpolation for position
            x = pose1[0] + t * (pose2[0] - pose1[0])
            y = pose1[1] + t * (pose2[1] - pose1[1])
            # Angular interpolation
            theta = normalizeAngle(pose1[2] + t * angle_diff)
            
            if not self.isValidPose([x, y, theta]):
                return False
        return True

    def extractPath(self):
        if self.goal.parent is None:
            return None
            
        # Get initial path
        path = []
        current = self.goal
        while current is not None:
            path.append(current.pose.tolist())
            current = current.parent
        path = path[::-1]  # Reverse to get start-to-goal order
        
        return path

class Node:
    def __init__(self, position, theta=0.0):
        self.position = position  # [x, y]
        self.theta = theta
        self.parent = None
        self.cost = 0.0

    @property
    def pose(self):
        return [*self.position, self.theta]

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
    
    def findValidTheta(self, position, from_node: Node):
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
        bounds = [(from_node.theta - self.max_dth, from_node.theta + self.max_dth)]
        
        # Optimize
        result = minimize(objective, theta0, bounds=bounds, method='SLSQP')
        
        if result.success:
            optimal_theta = result.x[0]
            # Verify the solution is actually valid
            robot_polygon = self.getRobotPolygon([position[0], position[1], optimal_theta])
            if not any(obs.intersects(robot_polygon) for obs in expanded_obstacles):
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
            optimal_theta = self.findValidTheta(new_pos, nearest_node)

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
                if distance_to_goal < self.step_size:
                    # Find optimal theta for connecting to goal
                    goal_theta = self.findValidTheta(self.goal.position, new_node)
                    
                    if (goal_theta is not None and 
                        self.isPathValid(new_node, self.goal.position, goal_theta)):
                        
                        self.goal.theta = goal_theta
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
            (0.3, 1.0),  # w2 range
            (0.2, 0.8)   # w3 range
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
                0.4 * norm_clearance + 
                0.2 * norm_orientation)

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

    def tune_parameters(self, n_calls=10):
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

    tuner = BayesianTuner(workspace_bounds, agent.pose, target_pose)
    best_params = tuner.tune_parameters()

    print(best_params)

    # while True:
    #     # rrt = RRTStar(
    #     #     start_pose=agent.pose,
    #     #     target_pose=target_pose,
    #     #     workspace_bounds=workspace_bounds,
    #     #     max_iter=5000,
    #     #     step_size=0.02,
    #     #     search_radius=1.0
    #     # )

    #     rrt = RRT(agent.pose, target_pose, workspace_bounds,
    #               w1=best_params['w1'], w2=best_params['w2'], w3=best_params['w3'])
        
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
    #         break  # Exit if no path is found

    # ---------------------------------------------------------------

