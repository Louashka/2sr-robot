import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import plotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from entities import robot_state, global_var as gv
from control import mm

class RobotAnimation:
    def __init__(self):
        self.robot = robot_state.Model(1, 0, 0, 0)
        self.target = [0.5, 0.5, np.pi/4, 0, 0]
        self.target_robot = robot_state.Model(2, *self.target)

        self.mm_controller = mm.MotionMorphologyControl(self.robot)

        self.rp = plotlib.RobotPlot()
        self.fig, self.ax = plt.subplots(figsize=(12, 12))

        self.fps = 15
        self.output_file = 'multimedia/motion_and_deformation.gif'

    def _update_robot_state(self):
        velocity, stiff_transitions, q_new, is_finished = self.mm_controller.go_to_target(self.target)
        print(f"Velocity: {velocity}")
        self.robot.config = q_new

    def _update_frame(self, frame_num: int):
        """The core function called for each frame of the animation."""
        # 1. Update the robot's state for the current frame
        self._update_robot_state()

        # 2. Clear the axes and redraw everything
        self.ax.clear()

        self.rp.plot_robot(self.ax, self.target_robot)
        self.rp.plot_robot(self.ax, self.robot)
        
        # 3. Set consistent plot limits and aspect ratio
        self.ax.set_aspect('equal')
        self.ax.set_xlim(-0.2, 0.7)
        self.ax.set_ylim(-0.2, 0.7)

    def run_and_save(self):
        """Creates the animation and saves it to a file."""
        ani = animation.FuncAnimation(
            fig=self.fig,
            func=self._update_frame,
            frames=100,
            interval=1000 / self.fps # Interval in milliseconds
        )

        print(f"Saving animation to {self.output_file}...")

        plt.tight_layout()
        writer = animation.PillowWriter(fps=self.fps )
        ani.save(self.output_file, writer=writer)
        print("...Done!")
        
        plt.close(self.fig)

if __name__ == "__main__":
    # Create the animation engine with the chosen scenario
    animator = RobotAnimation()

    # Run the process
    animator.run_and_save()