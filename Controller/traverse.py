from scipy.spatial import Voronoi
from scipy.optimize import minimize
from scipy.signal import find_peaks
from shapely.geometry import Polygon, Point, LineString
import numpy as np
import networkx as nx
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as PlotPolygon
from matplotlib.collections import PatchCollection
import pickle
import time
import func
import matplotlib.pyplot as plt
from matplotlib import animation
from View import plotlib
from Model import robot2sr
import robot2sr_controller as rsr_ctrl

voronoi_plot_path = 'Experiments/Figures/voronoi_plot.pkl'
agent_width = 0.07
agent_length = 0.33


# ----------------------------- Motion Planning -----------------------------

# Voronoi analysis 
class VoronoiPassageAnalyzer:
    def __init__(self, obstacles, workspace_bounds):
        self.obstacles = obstacles
        self.workspace_bounds = workspace_bounds
        self.boundary_points = self.sample_obstacle_boundaries()
        self.vor = Voronoi(self.boundary_points)
        self.passage_graph = self.create_passage_graph()
        self.allowed_passage_graph = None
        
    def sample_obstacle_boundaries(self):
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
    
    def identify_passages(self, agent_pos, min_clearance_threshold=0.5):
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
            distance = np.linalg.norm(np.array(node_pos) - np.array(agent_pos))
            if distance < min_distance:
                min_distance = distance
                closest_node = node
        
        # Get position of closest node
        closest_node_pos = self.allowed_passage_graph.nodes[closest_node]['pos']
        
        # Calculate direction vector from closest node to agent
        direction = np.array(agent_pos) - np.array(closest_node_pos)
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
            
            if np.linalg.norm(next_pos - np.array(agent_pos)) < step_size:
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
        print(f'Front point nodes: {front_nodes}\n')

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

def voronoiAnalysis(obstacles: list, initial_pose: list, target_pose: list) -> tuple[list, list, list, tuple]:
    bounds = [
        (-0.606, -0.8), # Lower left corner (x_min, y_min)
        (0.319, 0.5)  # Upper right corner (x_max, y_max)
    ]

    visualizer = VoronoiVisualizer(obstacles, bounds)
    
    # Plot basic Voronoi diagram
    fig, ax = visualizer.plot_voronoi(show_points=True)
    
    # Create analyzer and find passages
    analyzer = VoronoiPassageAnalyzer(obstacles, bounds)
    passages, _ = analyzer.identify_passages(initial_pose[:-1])

    # passages_list = [passage['nodes'] for passage in passages]
    # print(passages_list)
    
    # Highlight narrow passages
    visualizer.highlight_narrow_passages(passages)

    with open(voronoi_plot_path, 'wb') as file:
        pickle.dump({'figure': fig, 'axes': ax}, file)
    # plt.show()
    
    # Find passage sequence
    print('Find passage route...\n')
    rear_path_points, front_path_points, middle_path_points = analyzer.find_passage_sequence(initial_pose, target_pose, agent_length)

    front_path_points = [front_path_points_i.tolist() for front_path_points_i in front_path_points]
    middle_path_points = [middle_path_points_i.tolist() for middle_path_points_i in middle_path_points]
    rear_path_points = [rear_path_points_i.tolist() for rear_path_points_i in rear_path_points]

    return front_path_points, middle_path_points, rear_path_points, (fig, ax)


# Process Voronoi paths
def adjustPath(path_points, angle_threshold=2, max_merge_angle=15, max_zigzag_deviation=10) -> tuple[list, list]:
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
                new_path[prev_start + j] = (start_point + t[j] * (end_point - start_point)).tolist()
            
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
                new_path[start1 + j] = (start_point + t[j] * (end_point - start_point)).tolist()
            
            segment_starts.pop(i+1)
            segment_starts.pop(i)
        else:
            i += 1
    
    return segment_starts, new_path

def calcOrientations(middle_path, rear_path, connection_indices) -> list:
    theta_seq = []

    for i in range(1, len(middle_path)):
        moddle_point_vector = np.array([middle_path[i][0]-middle_path[i-1][0], 
                                        middle_path[i][1]-middle_path[i-1][1]])
        
        middle_rear_vector = np.array([rear_path[i][0]-middle_path[i-1][0], 
                                       rear_path[i][1]-middle_path[i-1][1]])

        angle = np.arctan2(moddle_point_vector[1], moddle_point_vector[0])

        if np.dot(moddle_point_vector, middle_rear_vector) < 0:
            angle = func.normalizeAngle(angle - np.pi)

        theta_seq.append(angle)
        
    theta_seq.append(theta_seq[-1])

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

            objective += 0.004 * (np.abs(q[3]) + np.abs(q[4]))
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


# Identify key robot configurations
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
    
    return [int(i) for i in filtered_indices]

def getCrucialConfigs(q_ref_path) -> list:
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
    configs_idx.append(len(q_ref_path) - 1)
    
    # Sort and remove duplicates
    configs_idx = sorted(list(set(configs_idx)))

    key_configs_idx = removeCloseIndices(configs_idx)

    # print(key_configs_idx)
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

    # axs[0].plot(key_configs_idx, q_ref[key_configs_idx, 3], 'ro')
    # axs[1].plot(key_configs_idx, q_ref[key_configs_idx, 4], 'ro')


    # plt.tight_layout()
    # plt.show()

    return key_configs_idx


# Animate motion planning results
def runAnimation(fig, rear_path_points, front_path_points, middle_path_points, theta_seq, q_ref, q_optim=None) -> None:
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
            if q_optim is not None:
                q_optim_i = q_optim[state.idx]
            
        if state.iteration == 0:
            # Iteration 0: Just show Voronoi diagram (empty plot_components)
            if state.counter > state.pause_frames_n:
                state.counter = 0
                state.iteration += 1
            else:
                state.counter += 1
        
        elif state.iteration == 1:
            # Iteration 1: rear point and its path only
            plotlib.plotPath(state.traversed_path, rear_pos, traversed_line, plot_components)
            rear_point.set_color(red_color)
            plotlib.plotPoints([rear_pos], (rear_point,), plot_components)

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
            plotlib.plotPath(state.traversed_path, rear_pos, traversed_line, plot_components)
            plotlib.plotPoints([front_pos, middle_pos, rear_pos], 
                        (front_point, middle_point, rear_point), plot_components)
            plotlib.plotOrientation(middle_pos, orientation, orientation_line, plot_components)

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
            plotlib.plotPoints([front_pos, middle_pos, rear_pos], 
                        (front_point, middle_point, rear_point), plot_components)
            plotlib.plotOrientation(middle_pos, orientation, orientation_line, plot_components)
            
            robot_ref_components = (vss1_line, vss2_line, conn1_line, conn2_line, 
                                    lu1_square, lu2_square)
            plotlib.plotRobot(q_ref_i, plot_components, robot_ref_components)

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

                    for component in robot_ref_components:
                        component.set_data([[], []])
                else:
                    state.counter += 1
        
        elif q_optim is not None:
            # robot_ref_components = (vss1_line, vss2_line, conn1_line, conn2_line, 
            #                         lu1_square, lu2_square)
            # for robot_ref_component in robot_ref_components:
            #     robot_ref_component.set_alpha(0.5)
            # plotlib.plotRobot(q_ref_i, plot_components, robot_ref_components)

            robot_optim_components = (vss1_line_optim, vss2_line_optim, conn1_line_optim, conn2_line_optim, 
                                      lu1_square_optim, lu2_square_optim)

            plotlib.plotRobot(q_optim_i, plot_components, robot_optim_components)

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
    
    plt.title('Voronoi Diagram with Robot Motion')
    plt.show()
    
# ----------------------------- Motion Control ------------------------------

# Control traversing obstacles
def discretizeConfigs(initial_config, way_points):
    target_configs = []
    initial_config = initial_config

    for way_point in way_points:
        num_intermediate = 2
        t = np.linspace(0, 1, num_intermediate + 2)

        # Special handling for angle (theta) to ensure proper interpolation
        # Unwrap theta to prevent interpolation issues around ±π
        theta_initial = initial_config[2]
        theta_target = way_point[2]
        delta_theta = (theta_target - theta_initial + np.pi) % (2 * np.pi) - np.pi
        theta_interp = theta_initial + delta_theta * t[1:-1]
        
        for i in range(num_intermediate):
            new_config = initial_config + (way_point - initial_config) * t[i+1]
            # Replace theta with properly interpolated angle
            new_config[2] = theta_interp[i]
            new_config = np.round(new_config, 4)
            target_configs.append(new_config.tolist())

        target_configs.append(np.round(way_point, 4).tolist())
        initial_config = way_point

    return target_configs

def traverseObstacles(agent: robot2sr.Robot, grasp_idx, target_configs, start_time, 
                      rgb_camera, simulation=False) -> tuple[dict, float]:
    agent_controller = rsr_ctrl.Controller()

    traverse_data = {}
    idx = 1

    for tc in target_configs:
        if idx == len(target_configs):
            rgb_camera.expanded_obstacles = []
            
        timestamps = []
        robot_states = []
        target_mm = []
        target_vel = []
        transition_status = []
        temperature_meas = []

        rgb_camera.target_robot_config = tc
        finish = False

        v_r = [0.0] * 3
        v_s = [0.0] * 2

        target_s = [0, 0]
        for i in range(2):
            if abs(tc[i+3] - agent.curvature[i]) > 5:
                target_s[i] = 1

        while True:
            elapsed_time = time.perf_counter() - start_time
            
            timestamps.append(elapsed_time)
            robot_states.append(agent.config.tolist())
            
            if target_s != [0, 0] and func.close2Shape(agent.curvature, tc[3:]):
                target_s = [0, 0]
                v_r = [0.0] * 3
                v_s = [0.0] * 2
            if (target_s == [0, 0] and func.close2Goal(agent.pose, tc[:3])) or rgb_camera.finish:
                v_r = [0.0] * 3
                v_s = [0.0] * 2
                finish = True
            else:
                if target_s == [0, 0]:
                    v_r, q_new = agent_controller.mpcRM(agent, tc[:3], v_r)
                elif target_s == [1, 0]:
                    v_s, q_new = agent_controller.mpcSM1(agent, tc, v_s)
                elif target_s == [0, 1]:
                    v_s, q_new = agent_controller.mpcSM2(agent, tc, v_s)
                else:
                    v_s, q_new = agent_controller.mpcSM3(agent, tc, v_s)

            v = v_r + v_s
            print(f'Stiffness: {target_s}')
            print(f'Velocity: {v}')

            target_mm.append(target_s)
            target_vel.append(v)

            _, _, current_s, sc_feedback = agent_controller.move(agent, v, target_s)
            print()

            if sc_feedback is None:
                transition_status.append(False)
            else:
                transition_status.append(True)

            temperature_meas.append(sc_feedback)
            
            if simulation:
                agent.config = q_new
                agent.stiffness = target_s
            else:
                agent.stiffness = current_s

            if finish:
                break

        elapsed_time = time.perf_counter() - start_time
            
        timestamps.append(elapsed_time)
        robot_states.append(agent.config.tolist())
        target_mm.append(target_s)
        target_vel.append(v)
        transition_status.append(False)
        temperature_meas.append(sc_feedback)
        
        traverse_data[idx] = {
            'target_config': tc,
            'tracking' : {
                'time': timestamps,
                'robot_states': robot_states,
                'target_mm': target_mm,
                'target_vel': target_vel,
                'transitions': transition_status,
                'temperature': temperature_meas
            }
        }
        idx += 1
        
        if rgb_camera.finish:
            break

        print('Next target!\n')

    return traverse_data, elapsed_time