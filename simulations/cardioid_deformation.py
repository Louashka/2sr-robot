import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import plotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from entities import robot_state, global_var as gv

plt.rc('font', size=28)   

class RobotAnimation:
    """
    An engine to create and save robot animations based on a cardioid
    scenario configuration.
    """
    def __init__(self, no: int):
        """
        Initializes the animator with a specific scenario.

        Args:
            no (int): A number of the cardioid to animate.
        """
        # --- Animation Scenarios ----
        self.ANIMATION_SCENARIOS = [
            {
                "name": "Cardioid 1",
                "cardioid_index": [1],
                "cardioid_color": ["k"],
                "animated_params": ["k2"],  
                "pose_change": False,
                "stiffness": [0, 1],
                "temp": [22, 64],
                "k_max": np.pi / gv.L_VSS,
                "cardioid_density": 65,
                "cardioid_size": 8, 
                "frame_count": 163,     
                "fps": 15,
                "output_file": "multimedia/cardioid1_animation.gif",
            },
            {
                "name": "Cardioid 2",
                "cardioid_index": [2, 1],
                "animated_params": ["k1"],
                "cardioid_color": ["k", "#338dc9"],
                "pose_change": True,
                "stiffness": [1, 0],
                "temp": [64, 22],
                "k_max": np.pi / (1.1 * gv.L_VSS),
                "cardioid_density": 65,
                "cardioid_size": 9,
                "frame_count": 163,
                "fps": 15,
                "output_file": "multimedia/cardioid2_animation.gif",
            },
            {
                "name": "Cardioid 3",
                "cardioid_index": [3, 1],
                "animated_params": ["k1", "k2"],
                "cardioid_color": ["k", "#338dc9"],
                "pose_change": True,
                "stiffness": [1, 1],
                "temp": [64, 64],
                "k_max": np.pi / (2 * gv.L_VSS),
                "cardioid_density": 65,
                "cardioid_size": 8,
                "frame_count": 163,
                "fps": 15,
                "output_file": "multimedia/cardioid3_animation.gif",
            }
        ]

        print(f"Initializing animation: {self.ANIMATION_SCENARIOS[no-1]['name']}")

        self.scenario = self.ANIMATION_SCENARIOS[no-1]

        # Setup the plot figure
        self.fig, self.ax = plt.subplots(figsize=(12, 11))
        self.fig.subplots_adjust(left=0.11, right=0.99, top=0.995, bottom=0.03)
        self.ax.set_aspect('equal')
        self.ax.set_xlim(-0.16, 0.29)
        self.ax.set_ylim(-0.22, 0.22)

        self.rp = plotlib.RobotPlot(self.ax)

        # --- Internal State ---
        self.robot = self._setup_initial_robot_state()
        self.animation_direction = 1

        # Calculate a smooth step size to complete a full cycle in the given frames
        self.k_step = (2 * self.scenario['k_max']) / ((self.scenario['frame_count']) / 2) 

        # Pre-calculate static plot elements
        self.cardioids = self._calculate_cardioid_path()

    def _setup_initial_robot_state(self) -> robot_state.Model:
        """Creates the initial robot model based on the scenario."""
        # Start with a straight robot to find the center point
        vss1_arc = self.rp.arc((0, 0, 0), 0)
        center = (vss1_arc[0][-1], vss1_arc[1][-1], vss1_arc[2])

        robot_model = robot_state.Model(
            id=1,
            x=center[0], y=center[1], theta=center[2],
            k1=0, k2=0,
            stiffness=self.scenario['stiffness']
        )
        robot_model.temp = self.scenario['temp']
        
        return robot_model

    def _calculate_cardioid_path(self) -> list:
        """Calculates the cardioid paths."""
        cardioid_paths = []
        for cardioid in self.scenario['cardioid_index']:
            cardioid_idx = cardioid - 1
            phi = np.linspace(0, 2 * np.pi, self.scenario['cardioid_density'])
            
            a = gv.CARDIOID_A[cardioid_idx]
            offset = gv.CARDIOID_OFFSET[cardioid_idx]
            
            # Parametric equations for the cardioid
            x = -2 * a * (1 - np.cos(phi)) * np.cos(phi) - offset + self.robot.x
            y =  2 * a * (1 - np.cos(phi)) * np.sin(phi) + self.robot.y
            
            if self.scenario['pose_change']:
                x -= gv.L_VSS

            cardioid_paths.append([x, y])

        return cardioid_paths

    def _update_frame(self, frame_num: int):
        """The core function called for each frame of the animation."""
        # 1. Update the robot's state for the current frame
        self._update_robot_state()

        # 2. Redraw the frame
        return self.rp.plot_robot(self.robot)

    def _update_robot_state(self):
        """Handles the logic for changing the animated parameter."""
        param_name = self.scenario['animated_params'][0]
        current_k = getattr(self.robot, param_name)
        
        # Reverse direction if the curvature limit is reached
        if abs(current_k) >= self.scenario['k_max']:
            self.animation_direction *= -1

        # Apply the change
        new_k = current_k + self.animation_direction * self.k_step

        for param_name in self.scenario['animated_params']:
            setattr(self.robot, param_name, new_k)

        if self.scenario['pose_change']:
            vss1_arc = self.rp.arc((0, 0, 0), new_k)
            center = (vss1_arc[0][-1], vss1_arc[1][-1], vss1_arc[2])

            self.robot.pose = center

    def run_and_save(self, save=False):
        """Creates the animation and saves it to a file."""
        for cardioid, clr in zip(self.cardioids, self.scenario['cardioid_color']):
            self.ax.plot(cardioid[0], cardioid[1], '.', color=clr, markersize=self.scenario['cardioid_size'])
        # self.ax.set_title(f"{self.scenario['name']}")

        ani = animation.FuncAnimation(
            fig=self.fig,
            func=self._update_frame,
            repeat=True, 
            blit=True,
            frames=self.scenario['frame_count'],
            interval=1000 / self.scenario['fps'] # Interval in milliseconds
        )

        if save:
            output_file = self.scenario['output_file']
            print(f"\nSaving animation to {output_file}...")

            writer = animation.PillowWriter(fps=self.scenario['fps'])
            ani.save(output_file, writer=writer)
            print("\n...Done!\n")
        
        plt.tight_layout()
        plt.show()
        plt.close(self.fig)


if __name__ == "__main__":
    # Choose which animation to run by name
    scenario_to_run = 3

    # Create the animation engine with the chosen scenario
    animator = RobotAnimation(scenario_to_run)

    # Run the process
    animator.run_and_save(save=True)


