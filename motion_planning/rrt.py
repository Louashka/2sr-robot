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
                 step_size=0.1, max_iter=5000, goal_tolerance=0.1):
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

    def _sample_config(self) -> np.ndarray:
        """Generates a random configuration within the defined bounds."""
        config = np.zeros(5)
        config[0] = np.random.uniform(self.config_bounds['x'][0], self.config_bounds['x'][1])
        config[1] = np.random.uniform(self.config_bounds['y'][0], self.config_bounds['y'][1])
        config[2] = np.random.uniform(self.config_bounds['theta'][0], self.config_bounds['theta'][1])
        config[3] = np.random.uniform(self.config_bounds['k1'][0], self.config_bounds['k1'][1])
        config[4] = np.random.uniform(self.config_bounds['k2'][0], self.config_bounds['k2'][1])
        return config

    def _steer(self, from_node: Node, to_config: np.ndarray) -> np.ndarray | None:
        """
        Moves a fixed step_size from a node towards a target configuration,
        checking for collisions along the entire edge.
        """
        direction_vector = to_config - from_node.config
        distance = np.linalg.norm(direction_vector)
        
        if distance == 0:
            return None
        
        direction_unit_vector = direction_vector / distance
        actual_step = min(self.step_size, distance)
        new_config = from_node.config + direction_unit_vector * actual_step

        num_interpolation_steps = 10
        intermediate_configs = np.linspace(from_node.config, new_config, num=num_interpolation_steps)[1:]

        for config in intermediate_configs:
            self.robot_model.config = config
            if is_in_collision(self.robot_model, self.env):
                return None 

        return new_config

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
        Executes the RRT algorithm to find a path.
        """
        tree = Tree(start_config)
        
        for i in range(self.max_iter):
            if np.random.rand() < 0.05:
                rand_config = goal_config
            else:
                rand_config = self._sample_config()
            
            nearest_node = tree.find_nearest_node(rand_config)
            new_config = self._steer(nearest_node, rand_config)
            
            if new_config is not None:
                new_node = tree.add_node(new_config, nearest_node)
                
                # --- MODIFIED LOGIC FOR GOAL CONNECTION ---
                goal_dist = np.linalg.norm(new_node.config[:2] - goal_config[:2])
                if goal_dist <= self.goal_tolerance:
                    print(f"Goal region reached in {i+1} iterations! Attempting final connection...")
                    
                    # Use our new dedicated connection function
                    final_config = self._connect_to_goal(new_node, goal_config)
                    
                    # The check is now much simpler: if the connection was successful, we're done.
                    if final_config is not None:
                        goal_node = tree.add_node(final_config, new_node)
                        final_path = tree.reconstruct_path(goal_node)
                        print("Successfully connected to the exact goal.")
                        return final_path, tree

        print("Failed to find a path within the maximum iterations.")
        return None, tree