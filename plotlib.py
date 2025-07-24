import matplotlib.axes
import numpy as np
import matplotlib.patches as patches
import matplotlib.transforms as transforms
import matplotlib.colors as mcolors
from entities import robot_state, global_var as gv

class RobotPlot:
    """
    A class dedicated to plotting a 2SR robot.
    The robot's physical structure is defined declaratively in KINEMATIC_CHAIN.
    """
    def __init__(self, ax) -> None:
        """
        Initializes the plotter and creates all necessary artists.

        Args:
            ax (matplotlib.axes.Axes): The axes on which to draw.
        """
        self.ax = ax
        # --- Style Configuration ---
        self.lw = 3
        self.lu_rounding_size = 0.005
        self.vss_flex_color = '#d6605e'
        self.fill_color = 'lightgrey'
        self.border_color = 'darkgrey'
        self.center_color = '#075B91'
        self.target_alpha = 0.4  # Transparency for the target plot

        # Temperature range for normalization and display
        self.max_temp = 63.0
        self.min_temp = 53.0
        self.room_temp = 22.0
        
        # Pre-convert colors to RGB for interpolation
        self.border_color_rgb = np.array(mcolors.to_rgb(self.border_color))
        self.vss_flex_color_rgb = np.array(mcolors.to_rgb(self.vss_flex_color))

        # --- Robot Structure Definition ---
        self.KINEMATIC_CHAIN = [
            {
                "id": 1,
                "direction": -1,  # VSS arc calculation direction
                "stiffness_attr": "stiff1",
                "kappa_attr": "k1",
                "temp_attr": "t1"
            },
            {
                "id": 2,
                "direction": 1,
                "stiffness_attr": "stiff2",
                "kappa_attr": "k2",
                "temp_attr": "t2"
            },
        ]

        # --- Artist Storage ---
        self.vss_lines, self.connector_patches, self.lu_patches = [], [], []
        self.vss_lines_target, self.connector_patches_target, self.lu_patches_target = [], [], []
        self.center_point = None
        
        # --- Artists for the Temperature Subplot ---
        self.temp_ax = None
        self.temp_bars = None

        self._init_artists()

    def _init_artists(self):
        """
        Creates all plotting artists once with dummy data.
        """
        # Create artists for each segment in the kinematic chain
        for _ in self.KINEMATIC_CHAIN:
            # Create a line artist for the VSS segment
            line, = self.ax.plot([], [], linewidth=self.lw)
            self.vss_lines.append(line)

            # Create a polygon patch for the connector triangle
            connector = patches.Polygon([[0,0], [0,0], [0,0]], closed=True, 
                                        facecolor=self.fill_color, 
                                        edgecolor=self.border_color, 
                                        linewidth=self.lw)
            self.ax.add_patch(connector)
            self.connector_patches.append(connector)

            # Create a FancyBboxPatch for the Locomotion Unit
            lu_patch = patches.FancyBboxPatch(
                xy=(-gv.LU_SIDE / 2, -gv.LU_SIDE / 2), width=gv.LU_SIDE, height=gv.LU_SIDE,
                boxstyle=f'round,pad=0,rounding_size={self.lu_rounding_size}',
                facecolor=self.fill_color, edgecolor=self.border_color, linewidth=self.lw
            )
            self.ax.add_patch(lu_patch)
            self.lu_patches.append(lu_patch)


            line, = self.ax.plot([], [], linewidth=self.lw, alpha=self.target_alpha)
            self.vss_lines_target.append(line)
            connector = patches.Polygon([[0,0], [0,0], [0,0]], closed=True, facecolor=self.fill_color, edgecolor=self.border_color, linewidth=self.lw, alpha=self.target_alpha)
            self.ax.add_patch(connector)
            self.connector_patches_target.append(connector)
            lu_patch = patches.FancyBboxPatch(xy=(-gv.LU_SIDE / 2, -gv.LU_SIDE / 2), width=gv.LU_SIDE, height=gv.LU_SIDE, boxstyle=f'round,pad=0,rounding_size={self.lu_rounding_size}', facecolor=self.fill_color, edgecolor=self.border_color, linewidth=self.lw, alpha=self.target_alpha)
            self.ax.add_patch(lu_patch)
            self.lu_patches_target.append(lu_patch)

        # Create an artist for the robot's center point
        self.center_point, = self.ax.plot([], [], 'o', color=self.center_color, zorder=10)

        # Initially, hide the target artists until they are needed
        self._set_target_visibility(False)

        # --- Create and Configure the Temperature Subplot ---
        fig = self.ax.get_figure()
        # Define position: [left, bottom, width, height] in figure coordinates (0-1)
        temp_ax_pos = [0.75, 0.13, 0.14, 0.07] 
        self.temp_ax = fig.add_axes(temp_ax_pos)
        
        # Create the horizontal bars. We plot T1 on top (y=1) and T2 on bottom (y=0).
        # Initialize with minimum temperature.
        y_pos = [1, 0] # y-coordinates for T1 and T2 bars
        initial_temps = [self.room_temp, self.room_temp]
        self.temp_bars = self.temp_ax.barh(y_pos, initial_temps, align='center', height=0.3)
        
        # Style the temperature subplot
        self.temp_ax.set_title("Temp (°C)", fontsize=10)
        self.temp_ax.set_xlim(self.room_temp, self.max_temp)
        self.temp_ax.set_yticks(y_pos)
        self.temp_ax.set_yticklabels(['T1', 'T2'])
        self.temp_ax.tick_params(axis='y', length=0) # Hide y-axis ticks
        self.temp_ax.tick_params(axis='x', labelsize=8)
        self.temp_ax.xaxis.grid(True, linestyle='--', alpha=0.6)
        
    def _set_target_visibility(self, visible: bool):
        """Helper function to show or hide all target artists at once."""
        all_target_artists = self.vss_lines_target + self.connector_patches_target + self.lu_patches_target
        for artist in all_target_artists:
            artist.set_visible(visible)

    def _get_stiffness_color(self, stiffness: bool, temperature: float, base_temp: float) -> np.ndarray:
        """
        Determines the appropriate color for a segment based on its stiffness state.
        
        - If not stiff, returns the rigid border color.
        - If stiff, interpolates between the border color and the flex color
          based on the provided temperature.

        Args:
            stiffness (bool): True if the segment is flexible/stiffening.
            temperature (float): The current temperature of the segment (22 to 63).

        Returns:
            np.ndarray: The calculated RGB color as a NumPy array.
        """       
        # 1. Normalize the temperature to a 0.0 to 1.0 factor.
        # We clip the value to handle any potential over/undershoots safely.
        temp_range = self.max_temp - base_temp
        if temp_range == 0: # Avoid division by zero
            factor = 1.0
        else:
            normalized_temp = (temperature - base_temp) / temp_range
            factor = np.clip(normalized_temp, 0.0, 1.0)

        # 2. Linearly interpolate between the two colors using the factor.
        # The formula is: C = C_start * (1 - f) + C_end * f
        interpolated_color = self.border_color_rgb * (1 - factor) + self.vss_flex_color_rgb * factor
        
        return interpolated_color
    
    def _update_temp_bars(self, robot: robot_state.Model):
        """Updates the temperature bars based on the robot's state."""
        # Data for the bars (T1, T2)
        temps = [getattr(robot, conf["temp_attr"]) for conf in self.KINEMATIC_CHAIN]
        stiffnesses = [getattr(robot, conf["stiffness_attr"]) for conf in self.KINEMATIC_CHAIN]

        # Update T1's bar (the first artist in self.temp_bars)
        self.temp_bars[0].set_width(temps[0])
        self.temp_bars[0].set_color(self._get_stiffness_color(stiffnesses[0], temps[0], self.room_temp))
        
        # Update T2's bar (the second artist in self.temp_bars)
        self.temp_bars[1].set_width(temps[1])
        self.temp_bars[1].set_color(self._get_stiffness_color(stiffnesses[1], temps[1], self.room_temp))
    
    def plot_robot(self, robot: robot_state.Model, target: robot_state.Model = None):
        """
        Main plotting method. Updates the main robot and, if provided, the
        transparent target robot.

        Args:
            robot (robot_state.Model): The current state of the robot.
            target (robot_state.Model, optional): The target state of the robot. 
                                                  If None, the target is hidden.
        """
        # Always update the main robot
        self._update_robot_artists(robot, self.vss_lines, self.connector_patches, self.lu_patches, self.center_point)

        # Update the target robot only if it's provided
        if target:
            self._set_target_visibility(True)
            self._update_robot_artists(target, self.vss_lines_target, self.connector_patches_target, self.lu_patches_target)
        else:
            self._set_target_visibility(False)

         # --- Update the temperature bars ---
        self._update_temp_bars(robot)

        # Return all artists that have been modified
        return tuple(self.vss_lines) + tuple(self.connector_patches) + tuple(self.lu_patches) + \
               tuple([self.center_point]) + tuple(self.vss_lines_target) + \
               tuple(self.connector_patches_target) + tuple(self.lu_patches_target) + \
               tuple(self.temp_bars)

    def _update_robot_artists(self, robot, vss_lines, connector_patches, lu_patches, center_point=None):
        """
        Generic internal function to update a set of robot artists based on a robot state.
        """
        for i, config in enumerate(self.KINEMATIC_CHAIN):
            stiffness = getattr(robot, config["stiffness_attr"])
            kappa = getattr(robot, config["kappa_attr"])
            
            vss_arc = self.arc(robot.pose, kappa, direction=config["direction"])
            
            # Update VSS line
            vss_lines[i].set_data(vss_arc[0], vss_arc[1])
            # vss_lines[i].set_color(self.vss_flex_color if stiffness else self.border_color)
            temperature = robot.temp[i] 
            # Get the dynamically calculated color
            dynamic_color = self._get_stiffness_color(stiffness, temperature, self.min_temp)
            # Set the color of the line
            vss_lines[i].set_color(dynamic_color)

            # Calculate geometry for connector and LU
            tip_xy = (vss_arc[0][-1], vss_arc[1][-1])
            base_xy = (tip_xy[0] + config["direction"] * gv.L_CONN * np.cos(vss_arc[2]), 
                       tip_xy[1] + config["direction"] * gv.L_CONN * np.sin(vss_arc[2]))

            # Update connector triangle
            v1, v2 = np.array(tip_xy), np.array(base_xy)
            perp_angle = vss_arc[2] - (np.pi / 2)
            v3 = v2 + (gv.LU_SIDE * 0.7) * np.array([np.cos(perp_angle), np.sin(perp_angle)])
            connector_patches[i].set_xy([v1, v2, v3])

            # Update LU
            lu_center = (base_xy[0] + (gv.LU_SIDE / 2) * (config["direction"] * np.cos(vss_arc[2]) + np.sin(vss_arc[2])),
                     base_xy[1] - (gv.LU_SIDE / 2) * (np.cos(vss_arc[2]) - config["direction"] * np.sin(vss_arc[2])))

            transform = (
                transforms.Affine2D().rotate(vss_arc[2]) +
                transforms.Affine2D().translate(lu_center[0], lu_center[1]) +
                self.ax.transData
            )
            lu_patches[i].set_transform(transform)
        
        # Update central point
        if center_point:
            center_point.set_data([robot.x], [robot.y])

    def arc(self, pose: list, k: float, direction=1) -> tuple[np.ndarray, np.ndarray, float]:
        """Calculates the arc curve of a VS segment"""
        l = np.linspace(0, gv.L_VSS, 50)
        theta_array = pose[2] + direction * k * l
        if abs(k) < 1e-9: # Robust check for straight line
            x = pose[0] + direction * l * np.cos(pose[2])
            y = pose[1] + direction * l * np.sin(pose[2])
        else:
            x = pose[0] + (np.sin(theta_array) - np.sin(pose[2])) / k
            y = pose[1] - (np.cos(theta_array) - np.cos(pose[2])) / k
        theta_end = theta_array[-1]
        return x, y, theta_end % (2 * np.pi)
        