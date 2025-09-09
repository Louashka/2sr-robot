import numpy as np
from shapely.geometry import Polygon, LineString

from entities import global_var as gv
from entities.robot_state import Model

def _calculate_arc(pose: list, k: float, direction: int, num: int = 20) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Calculates the arc curve of a VS segment.
    """
    l = np.linspace(0, gv.L_VSS, num)
    theta_array = pose[2] + direction * k * l
    if abs(k) < 1e-9:  # Straight line
        x = pose[0] + direction * l * np.cos(pose[2])
        y = pose[1] + direction * l * np.sin(pose[2])
    else:
        x = pose[0] + (np.sin(theta_array) - np.sin(pose[2])) / k
        y = pose[1] - (np.cos(theta_array) - np.cos(pose[2])) / k
    theta_end = theta_array[-1]
    return x, y, theta_end % (2 * np.pi)

def get_robot_body_polygons(robot: Model) -> dict[str, Polygon]:
    """
    Calculates the physical geometry of the robot as a set of polygons.

    Args:
        robot (Model): The robot's state object.

    Returns:
        dict[str, Polygon]: A dictionary mapping body part names to their
                            corresponding shapely Polygon objects.
    """
    body_parts = {}
    
    kinematic_chain = [
        {"id": 1, "direction": -1, "kappa_attr": "k1"},
        {"id": 2, "direction": 1, "kappa_attr": "k2"},
    ]

    current_pose = robot.pose
    
    for i, config in enumerate(kinematic_chain):
        kappa = getattr(robot, config["kappa_attr"])
        direction = config["direction"]
        
        # 1. Calculate the VSS segment as a LineString and buffer it to give it thickness
        vss_arc_x, vss_arc_y, theta_end = _calculate_arc(current_pose, kappa, direction)
        vss_line = LineString(np.column_stack([vss_arc_x, vss_arc_y]))
        # The buffer amount represents the radius of the segment
        body_parts[f'vss{i+1}'] = vss_line.buffer(0.01, cap_style=1) 

        # 2. Calculate the geometry for the connector and LU
        tip_xy = (vss_arc_x[-1], vss_arc_y[-1])
        base_xy = (tip_xy[0] + direction * gv.L_CONN * np.cos(theta_end),
                   tip_xy[1] + direction * gv.L_CONN * np.sin(theta_end))

        # 3. Create the LU polygon
        # Construct the four corners of the LU square based on its center and orientation
        lu_center_x = base_xy[0] + (gv.LU_SIDE / 2) * (direction * np.cos(theta_end) + np.sin(theta_end))
        lu_center_y = base_xy[1] - (gv.LU_SIDE / 2) * (np.cos(theta_end) - direction * np.sin(theta_end))
        
        angle = theta_end
        half_side = gv.LU_SIDE / 2
        
        # Define corners relative to center, then rotate and translate
        corners = np.array([
            [-half_side, -half_side],
            [ half_side, -half_side],
            [ half_side,  half_side],
            [-half_side,  half_side]
        ])
        
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle),  np.cos(angle)]
        ])
        
        rotated_corners = corners @ rotation_matrix.T
        transformed_corners = rotated_corners + np.array([lu_center_x, lu_center_y])
        
        body_parts[f'lu{i+1}'] = Polygon(transformed_corners)

    return body_parts