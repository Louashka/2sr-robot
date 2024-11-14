import sys
sys.path.append('D:/Robot 2SR/2sr-swarm-control')
from Model import global_var as gv, robot2sr as rsr, manipulandum, splines
import motive_client, robot2sr_controller as rsr_ctrl, camera_optitrack_synchronizer as cos
import threading
from datetime import datetime
from shapely.geometry import Polygon, Point, LineString
import cv2
import numpy as np

agent: rsr.Robot = None
object: manipulandum.Shape = None
obstacles = []

mocap = motive_client.MocapReader()
rgb_camera = cos.Aligner()

markers = None
agent_width = 0.06
agent_length = 0.33

original_obstacles = []
expanded_obstacles = []
target_pose = None

workspace_bounds = [-0.606, 0.319, -1.02, 1.034]

simulation = True

def updateConfig():
    global agent, object, markers

    agents_config, objects_config, markers, msg = mocap.getConfig()

    if agents_config and objects_config:
        agent_config = agents_config[0]
        if agent:
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

        if not expanded_obstacles and rgb_camera.obstacles is not None:
            expand_obstacles()

            rrt = RRTStar(
                start_pose=agent.pose,
                target_pose=target_pose,
                obstacles=original_obstacles,
                expanded_obstacles=expanded_obstacles,
                workspace_bounds=workspace_bounds,
                max_iter=2000,
                step_size=0.1,
                search_radius=1.5
            )

            path = rrt.plan()

            if path is not None:
                print("Path found!")
                print(target_pose)
                print(path)
                rgb_camera.rrt_path = path
            else:
                print("No path found!")

def expand_obstacles():
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
        worst_case_radius = agent_width/2
        expanded_poly = obstacle_poly.buffer(worst_case_radius, join_style=2)
        expanded_obstacles.append(expanded_poly)

        extended_corners = []
        xx, yy = expanded_poly.exterior.coords.xy
        for x, y in zip(xx.tolist()[:-1], yy.tolist()[:-1]):
            extended_corners.append((x, y))
        rgb_camera.expanded_obstacles_global.append(extended_corners)

def get_footprint(x, y, theta):
    """Get corner points of robot at given configuration"""
    # Calculate half dimensions
    w2 = agent_width / 2
    l2 = agent_length / 2
    
    # Define corners relative to center (counterclockwise)
    corners_local = np.array([
        [-l2, -w2],  # rear left
        [l2, -w2],   # front left
        [l2, w2],    # front right
        [-l2, w2]    # rear right
    ])
    
    # Rotation matrix
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    
    # Rotate corners and translate to position
    corners_global = np.dot(corners_local, R.T) + [x, y]
    return corners_global


class Node:
    def __init__(self, pose):
        self.pose = np.array(pose)  # [x, y, theta]
        self.parent = None
        self.cost = 0.0

class RRTStar:
    def __init__(self, start_pose, target_pose, obstacles, expanded_obstacles, 
                 workspace_bounds, max_iter=1000, step_size=0.5, search_radius=2.0):
        self.start = Node(start_pose)
        self.goal = Node(target_pose)
        self.obstacles = obstacles
        self.expanded_obstacles = expanded_obstacles
        self.bounds = workspace_bounds  # [x_min, x_max, y_min, y_max]
        self.max_iter = max_iter
        self.step_size = step_size
        self.search_radius = search_radius
        self.nodes = [self.start]

    def plan(self):
        for i in range(self.max_iter):
            # Sample random pose
            if np.random.random() < 0.05:  # 5% chance to sample goal
                sampled_pose = self.goal.pose
            else:
                sampled_pose = self.sample_random_pose()
            
            # Find nearest node
            nearest_node = self.get_nearest_node(sampled_pose)
            
            # Extend towards sampled pose
            new_pose = self.steer(nearest_node.pose, sampled_pose)
            
            # Check if new pose is valid
            if self.is_valid_pose(new_pose) and self.is_path_valid(nearest_node.pose, new_pose):
                new_node = Node(new_pose)
                
                # Find nearby nodes for rewiring
                nearby_nodes = self.get_nearby_nodes(new_pose)
                
                # Connect to best parent
                min_cost = float('inf')
                best_parent = None
                
                for node in nearby_nodes:
                    potential_cost = (node.cost + 
                                   self.calculate_distance(node.pose, new_pose))
                    
                    if (potential_cost < min_cost and 
                        self.is_path_valid(node.pose, new_pose)):
                        min_cost = potential_cost
                        best_parent = node
                
                if best_parent is not None:
                    new_node.parent = best_parent
                    new_node.cost = min_cost
                    self.nodes.append(new_node)
                    
                    # Rewire nearby nodes
                    self.rewire(new_node, nearby_nodes)
                    
                    # Check if we can connect to goal
                    if (self.calculate_distance(new_pose, self.goal.pose) < self.step_size and 
                        self.is_path_valid(new_pose, self.goal.pose)):
                        self.goal.parent = new_node
                        self.goal.cost = (new_node.cost + 
                                        self.calculate_distance(new_pose, self.goal.pose))
                        return self.extract_path()
        
        return None  # No path found
    
    def sample_random_pose(self):
        x = np.random.uniform(self.bounds[0], self.bounds[1])
        y = np.random.uniform(self.bounds[2], self.bounds[3])
        theta = np.random.uniform(-np.pi, np.pi)
        return np.array([x, y, theta])
    
    def get_nearest_node(self, pose):
        distances = [self.calculate_distance(node.pose, pose) for node in self.nodes]
        return self.nodes[np.argmin(distances)]
    
    def get_nearby_nodes(self, pose):
        nearby = []
        for node in self.nodes:
            if self.calculate_distance(node.pose, pose) <= self.search_radius:
                nearby.append(node)
        return nearby
    
    def steer(self, from_pose, to_pose):
        dist = self.calculate_distance(from_pose, to_pose)
        if dist <= self.step_size:
            return to_pose
        else:
            ratio = self.step_size / dist
            dx = to_pose[0] - from_pose[0]
            dy = to_pose[1] - from_pose[1]
            dtheta = self.normalize_angle(to_pose[2] - from_pose[2])
            
            new_x = from_pose[0] + dx * ratio
            new_y = from_pose[1] + dy * ratio
            new_theta = self.normalize_angle(from_pose[2] + dtheta * ratio)
            
            return np.array([new_x, new_y, new_theta])
    
    def rewire(self, new_node, nearby_nodes):
        for node in nearby_nodes:
            potential_cost = (new_node.cost + 
                            self.calculate_distance(new_node.pose, node.pose))
            
            if (potential_cost < node.cost and 
                self.is_path_valid(new_node.pose, node.pose)):
                node.parent = new_node
                node.cost = potential_cost
    
    def calculate_distance(self, pose1, pose2):
        # Euclidean distance for position, weighted angular difference
        pos_diff = np.sqrt((pose1[0] - pose2[0])**2 + (pose1[1] - pose2[1])**2)
        angle_diff = abs(self.normalize_angle(pose1[2] - pose2[2]))
        return pos_diff + 0.2 * angle_diff  # Weight for angular component
    
    def normalize_angle(self, angle):
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def is_valid_pose(self, pose):
        # Check if pose is within bounds and not in collision
        x, y = pose[0], pose[1]
        if not (self.bounds[0] <= x <= self.bounds[1] and 
                self.bounds[2] <= y <= self.bounds[3]):
            return False
            
        point = Point(x, y)
        return not any(obs.contains(point) for obs in self.expanded_obstacles)
    
    def is_path_valid(self, pose1, pose2, check_steps=5):
        # Check if path between poses is collision-free
        for i in range(check_steps):
            t = i / (check_steps - 1)
            x = pose1[0] + t * (pose2[0] - pose1[0])
            y = pose1[1] + t * (pose2[1] - pose1[1])
            if not self.is_valid_pose([x, y, 0]):
                return False
        return True
    
    def extract_path(self):
        if self.goal.parent is None:
            return None
            
        path = []
        current = self.goal
        while current is not None:
            path.append(current.pose.tolist())
            current = current.parent
        return path[::-1]  # Reverse to get start-to-goal order


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
    margin = 0.06

    for s in s_array:
        point = object.getPoint(s)
        theta = object.getTangent(s)

        vector_to_point = np.array(point) - object.position
        dot_product = np.dot(vector_to_point, direction_vector)
        
        if dot_product > max_dot_product:
            max_dot_product = dot_product
            point_with_margin = [point[0] + margin * np.cos(dir), 
                                 point[1] + margin * np.sin(dir)]
            target_pose = point_with_margin + [theta]

    rgb_camera.grasp_point = target_pose[:-1]  # Store the grasp point
    # ---------------------------------------------------------------

    # ------------------------ Path planning ------------------------
    '''
    ** Start first with path planning that does not require reshaping 
    of the robot 
    '''

    # ---------------------------------------------------------------

