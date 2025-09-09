import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from entities.robot_state import Model
from plotlib import RobotPlot
from entities import obstacles_map
from collision import is_in_collision
from motion_planning.rrt import RRT
from motion_planning.rrt_tree import Tree
from entities.obstacles_map import Environment

def phase0():
    """
    Main function to demonstrate Phase 0: Foundation.
    """
    print("Running Phase 0: Foundation Test...")

    # 1. SETUP THE ENVIRONMENT
    # Define some obstacles as shapely Polygons
    obstacle1 = Polygon([(0.2, 0.2), (0.2, 0.6), (0.3, 0.6), (0.3, 0.2)])
    obstacle2 = Polygon([(0.6, 0.5), (0.6, 0.8), (0.9, 0.8), (0.9, 0.5)])
    env = obstacles_map.Environment(obstacles=[obstacle1, obstacle2])

    # 2. SETUP THE PLOT
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.set_title("Phase 0: Robot Collision Checking")
    
    # Plot the environment obstacles
    env.plot(ax)

    # 3. INITIALIZE ROBOT AND PLOTTER
    # We will use two robot models: one for the current state, one for the target
    robot_current = Model(id=1, x=0.5, y=0.3, theta=0, k1=0, k2=0)
    robot_target = Model(id=2, x=0.0, y=0.0, theta=0, k1=0, k2=0) # Will be updated
    
    robot_plotter = RobotPlot(ax)

    # --- TEST CASE 1: SAFE POSITION ---
    robot_current.config = [0.5, 0.3, 1.57, 5, -5] # A safe, curved position
    
    # Check for collision
    is_colliding_safe = is_in_collision(robot_current, env)
    print(f"Robot at safe config is colliding: {is_colliding_safe}")
    
    # Plot the robot
    robot_plotter.plot_robot(robot_current)
    ax.text(robot_current.x, robot_current.y + 0.1, f"Collision: {is_colliding_safe}",
            color='green', ha='center')

    # --- TEST CASE 2: COLLIDING POSITION ---
    # Update the target robot to a configuration that should collide
    robot_target.config = [0.35, 0.4, 0.8, 5, 0]
    
    # Check for collision
    is_colliding_unsafe = is_in_collision(robot_target, env)
    print(f"Robot at target config is colliding: {is_colliding_unsafe}")

    # Plot the target robot state (will appear transparent)
    robot_plotter.plot_robot(robot_current, target=robot_target)
    ax.text(robot_target.x, robot_target.y + 0.1, f"Collision: {is_colliding_unsafe}",
            color='red', ha='center')

    plt.show()

def plot_tree(ax, tree: Tree):
    """Helper function to draw the RRT tree."""
    for node in tree.nodes:
        if node.parent:
            ax.plot([node.parent.config[0], node.config[0]],
                    [node.parent.config[1], node.config[1]],
                    color='lightblue', linewidth=0.5, zorder=1)

def plot_path(ax, path: list[np.ndarray]):
    """Helper function to draw the final path."""
    if not path:
        return
    path_coords = np.array(path)
    ax.plot(path_coords[:, 0], path_coords[:, 1], color='red', linewidth=2, zorder=2, label='Final Path')

def plot_tree(ax, tree: Tree):
    """Helper function to draw the RRT tree."""
    for node in tree.nodes:
        if node.parent:
            ax.plot([node.parent.config[0], node.config[0]],
                    [node.parent.config[1], node.config[1]],
                    color='lightblue', linewidth=0.5, zorder=1)

def plot_path(ax, path: list[np.ndarray]):
    """Helper function to draw the final path."""
    if not path:
        return
    path_coords = np.array(path)
    ax.plot(path_coords[:, 0], path_coords[:, 1], color='red', linewidth=2, zorder=2, label='Final Path')

def phase1():
    """Main function to demonstrate Phase 1: Basic RRT."""
    print("Running Phase 1: Basic RRT Test...")

    # 1. SETUP THE ENVIRONMENT
    obstacle = Polygon([(0.3, 0.2), (0.3, 0.8), (0.4, 0.8), (0.4, 0.2)])
    env = Environment(obstacles=[obstacle])

    # 2. DEFINE START AND GOAL
    start_config = np.array([0.1, 0.5, 0, 0, 0])
    goal_config = np.array([0.8, 0.5, 0, 0, 0])

    # 3. SETUP THE ALGORITHM
    config_bounds = {
        'x': [0, 1], 'y': [0, 1],
        'theta': [-np.pi, np.pi],
        'k1': [-10, 10], 'k2': [-10, 10]
    }
    
    # Create a robot model instance to be used by the algorithm
    robot_template = Model(id=99, x=0, y=0, theta=0)

    # Instantiate the RRT algorithm with its configuration
    rrt_algorithm = RRT(
        env=env,
        robot_model=robot_template,
        config_bounds=config_bounds,
        step_size=0.5,
        max_iter=500,
        goal_tolerance=0.2
    )

    # 4. RUN THE ALGORITHM
    final_path, tree = rrt_algorithm.run(start_config, goal_config)

    # 5. VISUALIZE THE RESULTS
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    ax.set_xlim(-0.05, 1)
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.set_title("Phase 1: Basic RRT Result")

    env.plot(ax)
    plot_tree(ax, tree)
    if final_path:
        plot_path(ax, final_path)

    robot_plotter = RobotPlot(ax)
    start_robot = Model(id=1, x=0,y=0,theta=0); start_robot.config = start_config
    goal_robot = Model(id=2, x=0,y=0,theta=0); goal_robot.config = goal_config
    robot_plotter.plot_robot(start_robot, target=goal_robot)
    
    ax.text(start_config[0], start_config[1] + 0.05, 'Start', color='green', ha='center')
    ax.text(goal_config[0], goal_config[1] + 0.05, 'Goal', color='purple', ha='center')
    
    ax.legend()
    plt.show()


if __name__ == '__main__':
    phase1()