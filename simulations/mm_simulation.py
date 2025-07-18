import sys
import os
import numpy as np
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import matplotlib.animation as animation
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import plotlib
from entities import robot_state, global_var as gv
from control import mm

class Simulation:
    """
    Encapsulates the logic for running a robot control simulation.
    
    Execute the control loop, update the robot's state, and record the history 
    of all relevant variables. 
    """
    def __init__(self, initial_robot_state: robot_state.Model):
        self.robot = initial_robot_state
        self.controller = mm.MMControl(self.robot)
        self.k_max = np.pi / (2 * gv.L_VSS)
        
        # A mapping for how stiffness actuators affect the robot state in simulation.
        self.STIFFNESS_UPDATE_MAP = {1: 0.5, -1: -0.5, 0: 0.0}

        self.history = self._initialize_history()

    def _initialize_history(self) -> Dict[str, List]:
        """Creates a dictionary to store all time-series data."""
        return {
            "targets": [], "states": [], "stiffness": [], "temp": [],
            "raw_velocities": [], "filtered_velocities": [],
            "stiffness_actions": []
        }

    def _append_to_history(self, target, raw_vel, filtered_vel, stiff_actions):
        """Appends the current step's data to the history dictionary."""
        self.history["targets"].append(target)
        self.history["states"].append(self.robot.config)
        self.history["temp"].append(self.robot.temp)
        self.history["stiffness"].append(self.robot.stiffness)
        self.history["raw_velocities"].append(raw_vel)
        self.history["filtered_velocities"].append(filtered_vel)
        self.history["stiffness_actions"].append(stiff_actions)

    def _apply_curvature_deadband(self, value: float) -> float:
        """
        Rounds the value if it falls within the deadband of (-2, 2).
        This snaps small curvature targets to zero.
        """
        if -2 < value < 2:
            return 0.0 
        else:
            return value

    def run(self, num_targets: int = 1):
        """
        Runs the full simulation for a specified number of random targets.

        Args:
            num_targets (int): The number of sequential targets to generate and pursue.

        Returns:
            Dict[str, List]: The populated history dictionary.
        """
        for i in range(num_targets):
            # Generate a new random target
            k1 = np.random.uniform(-self.k_max, self.k_max)
            k2 = np.random.uniform(-self.k_max, self.k_max)

            target_config = [
                self.robot.x + np.random.uniform(-0.5, 0.5), # Add to current pos for realism
                self.robot.y + np.random.uniform(-0.5, 0.5),
                self.robot.theta + np.random.uniform(-np.pi / 3, np.pi / 3),
                self._apply_curvature_deadband(k1),
                self._apply_curvature_deadband(k2)
            ]
            print(f"\n--- New Target ({i+1}/{num_targets}): {np.round(target_config, 2)} ---")
            self.controller.target = target_config

            # Record the initial state before starting the loop for this target
            self._append_to_history(target_config, [0.0] * 5, [0.0] * 5, [0, 0])

            is_finished = False
            while not is_finished:
                # Get commands from the controller
                raw_vel, filtered_vel, stiff_transitions, q_new, is_finished = self.controller.go_to_target()
                
                # Update robot state based on controller output
                self.robot.config = q_new
                self.robot.t1 += self.STIFFNESS_UPDATE_MAP.get(stiff_transitions[0], 0.0)
                self.robot.t2 += self.STIFFNESS_UPDATE_MAP.get(stiff_transitions[1], 0.0)

                print(f'Target: {[f'{v:.6f}' for v in target_config]}')
                print(f"Velocity: {[f'{v:.6f}' for v in raw_vel]}")
                print(f"State: {[f'{v:.6f}' for v in self.robot.config]}")
                print(f'Stiffness: {self.robot.stiffness}')

                self._append_to_history(target_config, raw_vel, filtered_vel, stiff_transitions)
        
        return self.history

class Visualization:
    """
    Handles all visualization tasks: animation and data plotting.
    """
    def __init__(self, results: Dict[str, List]):
        # Store data from the results dictionary
        self.target_history = np.array(results["targets"])
        self.state_history = np.array(results["states"])
        self.stiffness_history = results["stiffness"]
        self.temp_history = results["temp"]
        self.raw_velocity_history = np.array(results["raw_velocities"])
        self.filtered_velocity_history = np.array(results["filtered_velocities"])
        self.stiffness_actions_history = results["stiffness_actions"]

        self.frame_n = len(self.state_history)
        
        # Initialize robot models for plotting
        self.robot = robot_state.Model(1, *self.state_history[0])
        self.target_robot = robot_state.Model(2, *self.target_history[0])

        # --- Animation Setup ---
        self.fig_anim, self.ax_anim = plt.subplots(figsize=(12, 12))
        self.ax_anim.set_aspect('equal')
        self.robot_plotter = plotlib.RobotPlot(self.ax_anim)
        self.fps = 15
        self.output_file = 'multimedia/motion_and_deformation.gif'

        # --- Determine and set the plot limits ---
        xlim, ylim = self._determine_plot_limits()
        self.ax_anim.set_xlim(xlim)
        self.ax_anim.set_ylim(ylim)

        # --- Plot Configuration ---
        self.PLOT_CONFIG = [
            {
                "ax_pos": (0, 0), 
                "title": "Planar Velocities (vx, vy)", 
                "ylabel": "Velocity [m/s]",
                "plots": [
                    {"label": "v_x raw", "style": "--", "data": self.raw_velocity_history[:, 0]},
                    {"label": "v_y raw", "style": "--", "data": self.raw_velocity_history[:, 1]},
                    {"label": "v_x filtered", "data": self.filtered_velocity_history[:, 0]},
                    {"label": "v_y filtered",  "data": self.filtered_velocity_history[:, 1]},
                ]
            },
            {
                "ax_pos": (0, 1), 
                "title": "Angular Velocity (ω)", 
                "ylabel": "[rad/s]",
                "plots": [
                    {"label": "ω raw", "color": "purple", "style": "--", "data": self.raw_velocity_history[:, 2]},
                    {"label": "ω filtered", "color": "purple", "data": self.filtered_velocity_history[:, 2]},
                ]
            },
            {
                "ax_pos": (0, 2), 
                "title": "\"Soft\" Velocities (u1, u2)", 
                "ylabel": "[m/s]",
                "plots": [
                    {"label": "u1 raw", "style": "--", "data": self.raw_velocity_history[:, 3]},
                    {"label": "u2 raw", "style": "--", "data": self.raw_velocity_history[:, 4]},
                    {"label": "u1 filtered", "data": self.filtered_velocity_history[:, 3]},
                    {"label": "u2 filtered", "data": self.filtered_velocity_history[:, 4]},
                ]
            },
            {
                "ax_pos": (1, 0), 
                "title": "Robot Trajectory", 
                "xlabel": "x [m]", 
                "ylabel": "y [m]", 
                "aspect": "equal",
                "plots": [
                    {"label": "Path",  "style": ".-",
                     "data_x": self.state_history[:, 0], "data_y": self.state_history[:, 1]},
                     {"label": "Target", "style": "*", "markersize": 15,
                      "data_x": [self.target_history[-1, 0]], "data_y": [self.target_history[-1, 1]]},
                ]
            },
            {
                "ax_pos": (1, 1), 
                "title": "Orientation θ", 
                "ylabel": "Theta [rad]",
                "plots": [
                    {"label": "θ", "data": self.state_history[:, 2], "color": "orange"},
                    {"label": "Target θ", "color": "orange", "style": "--", "data": self.target_history[:, 2]},
                ]
            },
            {
                "ax_pos": (1, 2), 
                "title": "Curvature (k1, k2)", 
                "ylabel": "[m⁻¹]",
                "plots": [
                    {"label": "k1", "data": self.state_history[:, 3]},
                    {"label": "k2", "data": self.state_history[:, 4]},
                    {"label": "Target k1", "style": "--", "data": self.target_history[:, 3]},
                    {"label": "Target k2", "style": "--", "data": self.target_history[:, 4]},
                ]
            },
        ]

    def _determine_plot_limits(self) -> tuple:
        """
        Calculates optimal x and y axis limits to encompass the entire simulation.

        This method ensures the plot is square and includes a configurable padding 
        around the trajectory.

        Args:
            padding_factor (float): The percentage of the total range to add as
                                    padding (e.g., 0.1 for 10%).

        Returns:
            A tuple containing (xlim, ylim), where each is a (min, max) tuple.
        """
        # 1. Gather all x and y coordinates from both histories
        all_x = np.concatenate([self.state_history[:, 0], self.target_history[:, 0]])
        all_y = np.concatenate([self.state_history[:, 1], self.target_history[:, 1]])

        # 2. Find the global boundaries
        x_min, x_max = np.min(all_x), np.max(all_x)
        y_min, y_max = np.min(all_y), np.max(all_y)

        # 3. Calculate the range (width and height) of the data
        x_range = x_max - x_min
        y_range = y_max - y_min

        # 4. Determine the center of the data
        x_center = (x_max + x_min) / 2
        y_center = (y_max + y_min) / 2

        # 5. To maintain the 'equal' aspect ratio, find the largest range
        #    and use it for both width and height.
        max_range = max(x_range, y_range)
        
        # If there's no movement, prevent a zero range
        if max_range == 0:
            max_range = 1.0 # Default to a 1x1 area if static

        # 6. Calculate the padding and the final half-width of the plot
        plot_half_width = (max_range / 2) + 1.2 * gv.L_VSB

        # 7. Determine the final limits centered on the data
        xlim = (x_center - plot_half_width, x_center + plot_half_width)
        ylim = (y_center - plot_half_width, y_center + plot_half_width)

        return xlim, ylim

    def _update_animation_frame(self, frame_num: int):
        self.robot.config = self.state_history[frame_num]
        self.robot.stiffness = self.stiffness_history[frame_num]
        self.robot.temp = self.temp_history[frame_num]
        self.target_robot.config = self.target_history[frame_num]
        
        return self.robot_plotter.plot_robot(self.robot, self.target_robot)
    
    def run_animation(self, save_gif=False):
        """Creates and shows the animation, with an option to save."""
        ani = animation.FuncAnimation(
            fig=self.fig_anim, func=self._update_animation_frame,
            frames=self.frame_n, repeat=True, blit=True,
            interval=1000 / self.fps
        )
        
        if save_gif:
            print(f"Saving animation to {self.output_file}...")
            # Ensure the multimedia directory exists
            os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
            writer = animation.PillowWriter(fps=self.fps)
            ani.save(self.output_file, writer=writer)
            print("...Done!")
        
        plt.tight_layout()
        plt.show()
        plt.close(self.fig_anim)

    def plot_data(self):
        """Generates and shows data plots based on PLOT_CONFIG."""
        fig, axs = plt.subplots(2, 3, figsize=(20, 12))
        
        for config in self.PLOT_CONFIG:
            ax = axs[config["ax_pos"]]
            ax.set_title(config["title"])
            ax.set_xlabel(config.get("xlabel", "Timestep"))
            ax.set_ylabel(config.get("ylabel", ""))
            if "aspect" in config:
                ax.set_aspect(config["aspect"])
            
            for p in config["plots"]:
                # Handle both time-series and xy plots
                if "data_y" in p:
                    ax.plot(p["data_x"], p["data_y"], p.get("style", "-"), 
                            label=p["label"], color=p.get("color"), markersize=p.get("markersize"))
                else:
                    ax.plot(p["data"], p.get("style", "-"), 
                            label=p["label"], color=p.get("color"))
            
            ax.legend()
            ax.grid(True)

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # 1. Initialize the core components
    initial_robot = robot_state.Model(1, 0, 0, 0, 0, 0)
    initial_robot.t1 = 22  
    initial_robot.t2 = 22

    # 2. Run the simulation to generate data
    simulator = Simulation(initial_robot)
    simulation_results = simulator.run(num_targets=1)

    # 3. Pass the results to the visualizer
    visualizer = Visualization(simulation_results)

    # 4. Run the visualization tasks
    visualizer.run_animation()
    visualizer.plot_data()