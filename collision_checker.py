from entities.robot_state import Model
from entities import obstacles_map
import robot_geometry

def is_in_collision(robot: Model, env: obstacles_map.Environment) -> bool:
    """
    Checks if the robot is in a collision state with the environment.

    Args:
        robot (Model): The robot's current state.
        env (Environment): The environment containing the obstacles.

    Returns:
        bool: True if any part of the robot intersects with any obstacle,
              False otherwise.
    """
    # 1. Get the robot's current physical shape as a set of polygons.
    robot_body_parts = robot_geometry.get_robot_body_polygons(robot)

    # 2. Iterate through each obstacle in the environment.
    for obstacle in env.obstacles:
        # 3. Iterate through each part of the robot's body.
        for part_name, robot_part_poly in robot_body_parts.items():
            # 4. Use shapely's highly optimized intersection test.
            if robot_part_poly.intersects(obstacle):
                # For debugging, print which part hit which obstacle
                # print(f"Collision detected: {part_name} hit an obstacle.")
                return True # Found a collision, no need to check further.

    # If all loops finished without returning, there are no collisions.
    return False