import matplotlib.axes
import numpy as np
import matplotlib.patches as patches
import matplotlib.transforms as transforms
from entities import robot_state, global_var as gv

class RobotPlot:
    """
    A class dedicated to plotting a 2SR robot.
    The robot's physical structure is defined declaratively in KINEMATIC_CHAIN.
    """
    def __init__(self) -> None:
        # --- Style Configuration ---
        self.lw = 2
        self.lu_rounding_size = 0.005
        self.vss_flex_color = '#C44536'
        self.fill_color = 'lightgrey'
        self.border_color = 'darkgrey'
        self.center_color = '#2a9df4'

        # --- Robot Structure Definition ---
        self.KINEMATIC_CHAIN = [
            {
                "id": 1,
                "direction": -1,  # VSS arc calculation direction
                "stiffness_attr": "stiff1",
                "kappa_attr": "k1"
            },
            {
                "id": 2,
                "direction": 1,
                "stiffness_attr": "stiff2",
                "kappa_attr": "k2"
            },
        ]

    def plot_robot(self, ax: matplotlib.axes.Axes, robot: robot_state.Model):
        """
        Main plotting method that renders the robot based on the KINEMATIC_CHAIN definition.
        """
        # Loop through the robot structure and plot each segment assembly
        for segment_config in self.KINEMATIC_CHAIN:
            self._plot_segment_assembly(ax, robot, segment_config)

        # Plot the robot's central reference point.
        ax.plot(robot.x, robot.y, 'o', color=self.center_color, zorder=10)

    def _plot_segment_assembly(self, ax: matplotlib.axes.Axes, robot: robot_state.Model, config: dict):
        """
        Plots a complete segment assembly: VSS arc, connector, and LU.
        """
        # 1. Get segment-specific state using attributes from the config dict
        stiffness = getattr(robot, config["stiffness_attr"])
        kappa = getattr(robot, config["kappa_attr"])
        
        # 2. Calculate the VSS arc shape
        vss_arc = self._arc(robot.pose, kappa, direction=config["direction"])
        
        # 3. Plot the VSS arc with the correct color based on stiffness
        self._plot_vss_segment(ax, vss_arc, is_flexible=(stiffness == 1))

        # 4. Calculate connector and LU geometry
        tip_xy = (vss_arc[0][-1], vss_arc[1][-1])
        base_xy = (tip_xy[0] + config["direction"] * gv.L_CONN * np.cos(vss_arc[2]), 
                   tip_xy[1] + config["direction"] * gv.L_CONN * np.sin(vss_arc[2]))

        # 5. Plot the connector triangle
        self._plot_connector_triangle(
            ax, tip_xy, base_xy, vss_arc[2], 
            side_length=gv.LU_SIDE * 0.7
        )

        # 6. Plot the Locomotion Unit
        lu_center = (base_xy[0] + (gv.LU_SIDE / 2) * (config["direction"] * np.cos(vss_arc[2]) + np.sin(vss_arc[2])),
                     base_xy[1] - (gv.LU_SIDE / 2) * (np.cos(vss_arc[2]) - config["direction"] * np.sin(vss_arc[2])))
        self._plot_lu(ax, lu_center, vss_arc[2], gv.LU_SIDE)

    def _arc(self, pose: list, k: float, direction=1) -> tuple[np.ndarray, np.ndarray, float]:
        """Calculates the arc curve of a VS segment"""
        l = np.linspace(0, gv.L_VSS, 50)
        theta_array = pose[2] + direction * k * l
        if abs(k) < 1e-9: # Robust check for straight line
            x = pose[0] + direction * l * np.cos(pose[2])
            y = pose[1] + direction * l * np.sin(pose[2])
        else:
            x = pose[0] + (np.sin(theta_array) - np.sin(pose[2])) / (direction * k)
            y = pose[1] - (np.cos(theta_array) - np.cos(pose[2])) / (direction * k)
        theta_end = theta_array[-1]
        return x, y, theta_end % (2 * np.pi)

    def _plot_vss_segment(self, ax: matplotlib.axes.Axes, vss_arc_data: tuple, is_flexible: bool):
        """Plots the VSS arc with the correct color."""
        color = self.vss_flex_color if is_flexible else self.border_color
        ax.plot(vss_arc_data[0], vss_arc_data[1], color=color, lw=self.lw) 
    
    def _plot_connector_triangle(self, ax: matplotlib.axes.Axes, tip_xy: tuple, base_xy: tuple, angle_rad: float, side_length: float):
        """
        Plots a VSS connector as a filled, right-angle triangle.

        The right angle is located at the base of the connector, where it
        meets the Locomotion Unit.

        Args:
            ax: The matplotlib axes to plot on.
            tip_xy: The (x, y) coordinate of the triangle's tip (at the VSS end).
            base_xy: The (x, y) coordinate of the right-angle vertex (at the LU end).
            angle_rad: The orientation angle of the connector's main line.
            side_length: The length of the side perpendicular to the connector line.
        """
        v1, v2 = np.array(tip_xy), np.array(base_xy)
        perp_angle = angle_rad - (np.pi / 2)
        v3 = v2 + side_length * np.array([np.cos(perp_angle), np.sin(perp_angle)])

        triangle = patches.Polygon(
            [v1, v2, v3], closed=True, facecolor=self.fill_color,
            edgecolor=self.border_color, linewidth=self.lw
        )

        ax.add_patch(triangle)
    
    def _plot_lu(self, ax: matplotlib.axes.Axes, center_xy: tuple, angle_rad: float, size: float):
        """Plots a single, rotated Locomotion Unit as a rounded square."""
        
        boxstyle = f'round,pad=0,rounding_size={self.lu_rounding_size}'

        lu_patch = patches.FancyBboxPatch(
            xy=(-size / 2, -size / 2), width=size, height=size, boxstyle=boxstyle,
            facecolor=self.fill_color, edgecolor=self.border_color, linewidth=self.lw
        )

        transform = (
            transforms.Affine2D().rotate(angle_rad) +
            transforms.Affine2D().translate(center_xy[0], center_xy[1]) +
            ax.transData
        )

        lu_patch.set_transform(transform)

        ax.add_patch(lu_patch)