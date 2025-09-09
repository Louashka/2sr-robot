import numpy as np

class Node:
    """
    Represents a single node in the RRT search tree.
    """
    def __init__(self, config: np.ndarray, parent=None):
        """
        Initializes a Node.

        Args:
            config (np.ndarray): The robot's configuration [x, y, theta, k1, k2].
            parent (Node, optional): The parent node in the tree. Defaults to None for the root.
        """
        self.config = config
        self.parent = parent
        # Cost is the cumulative path length from the root to this node.
        self.cost = 0.0

class Tree:
    """
    Represents the RRT search tree itself.
    """
    def __init__(self, root_config: np.ndarray):
        """
        Initializes the Tree with a root node.

        Args:
            root_config (np.ndarray): The starting configuration of the robot.
        """
        self.root = Node(root_config)
        self.nodes = [self.root]

    def add_node(self, config: np.ndarray, parent: Node) -> Node:
        """
        Creates a new node and adds it to the tree.

        Args:
            config (np.ndarray): The configuration for the new node.
            parent (Node): The parent node to link to.

        Returns:
            Node: The newly created and added node.
        """
        new_node = Node(config, parent)
        new_node.cost = parent.cost + np.linalg.norm(new_node.config[:2] - parent.config[:2])
        self.nodes.append(new_node)
        return new_node


    def find_nearest_node(self, target_config: np.ndarray) -> Node:
        """
        Finds the node in the tree that is closest to a target configuration.

        Args:
            target_config (np.ndarray): The configuration to search near.

        Returns:
            Node: The node in the tree with the minimum distance to the target.
        """
        distances = [np.linalg.norm(node.config - target_config) for node in self.nodes]
        nearest_index = np.argmin(distances)
        return self.nodes[nearest_index]

    def find_neighbors_in_radius(self, config: np.ndarray, radius: float) -> list[Node]:
        """Finds all nodes within a given radius of a configuration."""
        neighbors = []
        for node in self.nodes:
            # We only care about x,y distance for neighbor search
            if np.linalg.norm(node.config[:2] - node.config[:2]) <= radius:
                neighbors.append(node)
        return neighbors

    def reconstruct_path(self, goal_node: Node) -> list[np.ndarray]:
        """
        Backtracks from a goal node to the root to find the final path.

        Args:
            goal_node (Node): The node that successfully reached the goal region.

        Returns:
            list[np.ndarray]: A list of configurations representing the path
                              from start to goal.
        """
        path = []
        current_node = goal_node
        while current_node is not None:
            path.append(current_node.config)
            current_node = current_node.parent
        return path[::-1] # Reverse the path to go from start to goal