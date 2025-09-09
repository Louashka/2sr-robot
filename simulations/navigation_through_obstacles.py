import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from entities.robot_state import Model
from entities.coordinate_frame import Frame # Assuming this exists
from plotlib import RobotPlot
from entities import obstacles_map
from collision_checker import is_in_collision

def main():
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


if __name__ == '__main__':
    # You might need to add the project root to your Python path
    # if you have issues with imports, e.g., from entities.
    # For now, we assume you run this from the project_folder.
    main()