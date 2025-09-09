import numpy as np
from motion_planning.rrt_tree import Tree, Node
from entities.obstacles_map import Environment
from collision import is_in_collision 
from entities.robot_state import Model

class RRT:
    """
    Contains the logic for the basic RRT algorithm.
    This class is configured with the environment and algorithm parameters and
    can then be used to compute a path.
    """
    def __init__(self, env: Environment, robot_model: Model, config_bounds: dict, 
                 step_size=0.1, max_iter=5000, goal_tolerance=0.1, search_radius=0.3):
        """
        Initializes the RRT algorithm configuration.

        Args:
            env (Environment): The environment with obstacles.
            robot_model (Model): A template robot model instance used for collision checks.
            config_bounds (dict): Min/max for each config variable, e.g., {'x': [0, 1], ...}.
            step_size (float): The fixed distance to extend the tree (epsilon).
            max_iter (int): The maximum number of iterations.
            goal_tolerance (float): The radius around the goal to consider it "reached".
        """
        self.env = env
        self.robot_model = robot_model
        self.config_bounds = config_bounds
        self.step_size = step_size
        self.max_iter = max_iter
        self.goal_tolerance = goal_tolerance
        self.search_radius = search_radius

    def _sample_config(self) -> np.ndarray:
        """Generates a random configuration within the defined bounds."""
        config = np.zeros(5)
        config[0] = np.random.uniform(self.config_bounds['x'][0], self.config_bounds['x'][1])
        config[1] = np.random.uniform(self.config_bounds['y'][0], self.config_bounds['y'][1])
        config[2] = np.random.uniform(self.config_bounds['theta'][0], self.config_bounds['theta'][1])
        config[3] = self.robot_model.k1
        config[4] = self.robot_model.k2
        return config

    def _steer(self, from_config: np.ndarray, to_config: np.ndarray) -> np.ndarray | None:
        """
        Moves a fixed step_size from a config towards a target config.
        This version does NOT do collision checking; it only computes the new point.
        """
        direction_vector = to_config - from_config
        distance = np.linalg.norm(direction_vector)
        
        if distance == 0:
            return from_config
        
        # Move step_size towards the target, or directly to it if it's closer.
        actual_step = min(self.step_size, distance)
        new_config = from_config + (direction_vector / distance) * actual_step
        return new_config

    def _is_edge_collision_free(self, from_config: np.ndarray, to_config: np.ndarray) -> bool:
        """
        --- NEW HELPER FOR RRT* ---
        Checks if a straight-line path between two configurations is collision-free.
        This is crucial for checking potential new connections during rewiring.
        """
        # Use a fine resolution for checking edges
        num_interpolation_steps = int(np.linalg.norm(to_config[:2] - from_config[:2]) / 0.02)
        if num_interpolation_steps < 2:
            num_interpolation_steps = 2
            
        intermediate_configs = np.linspace(from_config, to_config, num=num_interpolation_steps)

        for config in intermediate_configs[1:]: # Start from the second point
            self.robot_model.config = config
            if is_in_collision(self.robot_model, self.env):
                return False
        return True

    def _connect_to_goal(self, from_node: Node, goal_config: np.ndarray) -> np.ndarray | None:
        """
        Attempts to connect a node directly to the goal configuration,
        performing collision checking along the entire path. This method is
        not limited by step_size.
        """
        # We perform the same discretized collision check as _steer, but
        # the destination is the actual goal, not a point one step away.
        num_interpolation_steps = 20 # Use more steps for this critical final connection
        
        intermediate_configs = np.linspace(from_node.config, goal_config, num=num_interpolation_steps)[1:]

        for config in intermediate_configs:
            self.robot_model.config = config
            if is_in_collision(self.robot_model, self.env):
                # The direct path to the goal is obstructed.
                return None
        
        # If the loop completes, the path is clear. Return the goal configuration.
        return goal_config

    def run(self, start_config: np.ndarray, goal_config: np.ndarray) -> tuple[list[np.ndarray] | None, Tree]:
        """
        Executes the RRT* algorithm to find an optimal path.
        """
        tree = Tree(start_config)
        best_goal_node = None # --- RRT* tracks the best goal found so far

        for i in range(self.max_iter):
            # 1. SAMPLE A RANDOM CONFIGURATION
            if np.random.rand() < 0.05:
                rand_config = goal_config
            else:
                rand_config = self._sample_config()
            
            # 2. FIND THE NEAREST NODE IN THE TREE
            nearest_node = tree.find_nearest_node(rand_config)
            
            # 3. STEER FROM NEAREST NODE TO THE RANDOM CONFIG
            new_config = self._steer(nearest_node.config, rand_config)
            
            # 4. CHOOSE BEST PARENT (RRT* Core Logic)
            # Find all neighbors within the search radius
            neighbors = tree.find_neighbors_in_radius(new_config, self.search_radius)
            if not neighbors:
                neighbors.append(nearest_node) # Fallback if radius is too small

            # Find the neighbor that results in the lowest cost path to new_config
            best_parent = nearest_node
            min_cost = nearest_node.cost + np.linalg.norm(new_config[:2] - nearest_node.config[:2])

            for neighbor in neighbors:
                potential_cost = neighbor.cost + np.linalg.norm(new_config[:2] - neighbor.config[:2])
                # If this neighbor offers a cheaper path AND the edge is collision-free...
                if potential_cost < min_cost and self._is_edge_collision_free(neighbor.config, new_config):
                    min_cost = potential_cost
                    best_parent = neighbor
            
            # If the best connection is not collision-free, skip this iteration
            # This check is implicitly handled by the next step's _is_edge_collision_free
            # but we can do it explicitly for clarity.
            if not self._is_edge_collision_free(best_parent.config, new_config):
                continue

            # 5. ADD THE NEW NODE TO THE TREE
            new_node = tree.add_node(new_config, best_parent)

            # 6. REWIRE THE TREE (RRT* Core Logic)
            # For each neighbor, check if its path is shorter by connecting through new_node
            for neighbor in neighbors:
                if neighbor == best_parent:
                    continue
                
                potential_new_cost = new_node.cost + np.linalg.norm(neighbor.config[:2] - new_node.config[:2])
                
                if potential_new_cost < neighbor.cost and self._is_edge_collision_free(new_node.config, neighbor.config):
                    neighbor.parent = new_node
                    neighbor.cost = potential_new_cost # Update the cost of the rewired node and its descendants

            # 7. CHECK FOR GOAL
            goal_dist = np.linalg.norm(new_node.config[:2] - goal_config[:2])
            if goal_dist <= self.goal_tolerance:
                # If we can connect to the goal, we have found a potential solution
                if self._is_edge_collision_free(new_node.config, goal_config):
                    print(f"Goal region reached at iteration {i+1}. Evaluating path...")
                    
                    # Create a temporary node for the exact goal
                    final_node = Node(goal_config, new_node)
                    final_node.cost = new_node.cost + np.linalg.norm(final_node.config[:2] - new_node.config[:2])

                    # If this is the first time we reach the goal, or if this path is cheaper
                    if best_goal_node is None or final_node.cost < best_goal_node.cost:
                        best_goal_node = final_node
                        print(f"Found a new, better path to the goal with cost: {best_goal_node.cost:.2f}")

        # After all iterations, if we ever found a path to the goal, reconstruct it
        if best_goal_node:
            # We add the final goal node to the tree for visualization purposes
            tree.nodes.append(best_goal_node)
            final_path = tree.reconstruct_path(best_goal_node)
            print("Finished search. Returning best path found.")
            return final_path, tree

        print("Failed to find a path within the maximum iterations.")
        return None, tree