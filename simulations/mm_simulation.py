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

        """
        This map contains the complete logic for stiffness updates.

        - Command 1:  Always heats (returns +0.5).
        - Command -1: Always cools (returns -0.25). The cooling rate is slower than the 
                      the heating rate.
        - Command 0:  Conditional hold/cool.
                      - If stiffness == 1, holds (returns 0.0).
                      - Else if temp > 22, cools towards the minimum (returns -0.25).
        """
        self.STIFFNESS_LOGIC_MAP = {
            # Command 1: Always returns 0.5, ignoring state.
            1: lambda stiff, temp: 0.5,

            # Command -1: Always returns -0.25, ignoring state.
            -1: lambda stiff, temp: -0.25,

            # Command 0: Contains the conditional logic.
            0: lambda stiff, temp: (
                0.0 if stiff else 
                -0.25 if temp > 22.25 else 
                0.0
            )
        }

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
        
    def _pursue_target(self, target_config: List[float], target_label: str):
        """
        Executes the control loop to reach a single specified target.

        This method encapsulates the logic for setting a target, running the
        controller until the target is reached, and recording the history
        at each step.

        Args:
            target_config (List[float]): The target state [x, y, th, k1, k2].
            target_label (str): A descriptive string for logging purposes.
        """
        print(f"\n--- Pursuing {target_label}: {np.round(target_config, 2)} ---")
        self.controller.target = target_config

        # Record the state *before* starting the pursuit of this new target
        self._append_to_history(target_config, [0.0] * 5, [0.0] * 5, [0, 0])

        is_finished = False
        while not is_finished:
            # Get commands from the controller
            # NOTE: Assuming go_to_target is updated to return raw_vel as well
            raw_vel, filtered_vel, stiff_transitions, q_new, is_finished = self.controller.go_to_target()
            
            # Update robot state based on controller output
            self.robot.config = q_new
            self.robot.t1 += self.STIFFNESS_LOGIC_MAP.get(stiff_transitions[0])(self.robot.stiff1, self.robot.t1)
            self.robot.t2 += self.STIFFNESS_LOGIC_MAP.get(stiff_transitions[1])(self.robot.stiff2, self.robot.t2)

            # Logging for the current step
            print(f"Target      : {[f'{v:.6f}' for v in target_config]}")
            print(f"Velocity    : {[f'{v:.6f}' for v in raw_vel]}")
            print(f"State       : {[f'{v:.6f}' for v in self.robot.config]}")
            print(f"Stiffness   : {self.robot.stiffness}")

            self._append_to_history(target_config, raw_vel, filtered_vel, stiff_transitions)

    def run(self, num_targets: int = 1):
        """
        Runs the full simulation for a specified number of random targets.

        If num_targets > 1, the robot will visit all generated targets and then
        return to its original starting position.

        Args:
            num_targets (int): The number of sequential targets to generate and pursue.

        Returns:
            Dict[str, List]: The populated history dictionary.
        """
        # --- Phase 1: Capture initial state and generate target sequence ---

        # 1. Store the robot's absolute starting configuration. Use .copy()!
        initial_robot_config = self.robot.config.copy().tolist()
        print(f"INFO: Robot starting at: {np.round(initial_robot_config, 2)}.")

        targets_to_pursue = []
        for _ in range(num_targets):
            k1 = round(np.random.uniform(-self.k_max, self.k_max), 3)
            k2 = round(np.random.uniform(-self.k_max, self.k_max), 3)
            
            # Base the next target's position on the last point in the sequence
            # (or the initial position if the sequence is empty).
            base_pos = targets_to_pursue[-1] if targets_to_pursue else initial_robot_config

            target_config = [
                base_pos[0] + np.random.uniform(-0.5, 0.5),
                base_pos[1] + np.random.uniform(-0.5, 0.5),
                base_pos[2] + np.random.uniform(-np.pi / 3, np.pi / 3),
                self._apply_curvature_deadband(k1),
                self._apply_curvature_deadband(k2)
            ]
            targets_to_pursue.append(target_config)

        # 2. If it's a multi-target mission, add the initial position as the final target.
        if num_targets > 1:
            targets_to_pursue.append(initial_robot_config)
            print("INFO: Appending initial position to create a return-to-home path.")

        # --- Phase 2: Execute the planned sequence ---
        for i, target in enumerate(targets_to_pursue):
            # Create a descriptive label for logging
            is_return_home_target = (i == len(targets_to_pursue) - 1) and (num_targets > 1)
            label = f"Return to Initial Position (Target {i+1}/{len(targets_to_pursue)})" if is_return_home_target else f"Target {i+1}/{len(targets_to_pursue)}"
            
            self._pursue_target(target, label)
        
        print("\n--- Simulation Finished ---")

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
        self.target_robot.temp = [22, 22]

        # --- Animation Setup ---
        self.fig_anim, self.ax_anim = plt.subplots(figsize=(12, 12))
        self.ax_anim.set_aspect('equal')
        self.robot_plotter = plotlib.RobotPlot(self.ax_anim)
        self.fps = 15
        self.output_file = 'multimedia/motion_and_deformation.mp4'

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
    
    def run_animation(self, save=False):
        """Creates and shows the animation, with an option to save."""
        ani = animation.FuncAnimation(
            fig=self.fig_anim, func=self._update_animation_frame,
            frames=self.frame_n, repeat=True, blit=True,
            interval=1000 / self.fps
        )
        
        if save:
            print(f"Saving animation to {self.output_file}...")
            # Ensure the multimedia directory exists
            os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
            # writer = animation.PillowWriter(fps=self.fps)
            ani.save(self.output_file, writer='ffmpeg')
            print("...Done!")
        
        plt.tight_layout()
        plt.show()
        plt.close(self.fig_anim)

    def plot_data(self):
        """Generates and shows data plots based on PLOT_CONFIG."""
        fig, axs = plt.subplots(2, 3, figsize=(16, 10))
        
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
    initial_robot = robot_state.Model(1, 1, -2, np.pi/4, 10, 0)
    initial_robot.t1 = 22  
    initial_robot.t2 = 22

    # 2. Run the simulation to generate data
    simulator = Simulation(initial_robot)
    simulation_results = simulator.run(num_targets=1)

    # 3. Pass the results to the visualizer
    visualizer = Visualization(simulation_results)

    # 4. Run the visualization tasks
    visualizer.run_animation(save=True)
    visualizer.plot_data()