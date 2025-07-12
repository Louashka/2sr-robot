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
    def __init__(self, target_robot: robot_state.Model, state_history, stiffness_history, velocity_history, stiffness_actions_history):
        self.target_robot = target_robot
        self.state_history = state_history
        self.stiffness_history = stiffness_history
        self.velocity_history = velocity_history
        self.stiffness_actions_history = stiffness_actions_history

        self.frame_n = len(self.state_history)
        
        self.robot = robot_state.Model(1, *state_history[0])
        self.mm_controller = mm.MotionMorphologyControl(self.robot)

        self.rp = plotlib.RobotPlot()
        self.fig, self.ax = plt.subplots(figsize=(12, 12))

        self.fps = 15
        self.output_file = 'multimedia/motion_and_deformation.gif'

    def _update_frame(self, frame_num: int):
        """The core function called for each frame of the animation."""
        # 1. Update the robot's state for the current frame
        self.robot.config = self.state_history[frame_num]
        self.robot.stiffness = self.stiffness_history[frame_num]

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
        self.ani = animation.FuncAnimation(
            fig=self.fig,
            func=self._update_frame,
            frames=self.frame_n,
            interval=1000 / self.fps # Interval in milliseconds
        )

        print(f"Saving animation to {self.output_file}...")

        plt.tight_layout()
        writer = animation.PillowWriter(fps=self.fps )
        self.ani.save(self.output_file, writer=writer)
        print("...Done!")
        
        plt.close(self.fig)

    def plot_data(self):
        # Plot all histories as subplots in a single figure
        fig, axs = plt.subplots(2, 3, figsize=(18, 12))

        state_history = np.array(self.state_history)
        velocity_history = np.array(self.velocity_history)

        # 1) vx_history and vy_history
        axs[0, 0].plot(velocity_history[:,0], label='v_x')
        axs[0, 0].plot(velocity_history[:,1], label='v_y')
        axs[0, 0].set_title('vx and vy History')
        axs[0, 0].set_xlabel('Timestep')
        axs[0, 0].set_ylabel('Velocity (m/s)')
        axs[0, 0].legend()
        axs[0, 0].grid(True)

        # 2) omega_history
        axs[0, 1].plot(velocity_history[:,2], label='omega', color='purple')
        axs[0, 1].set_title('Omega History')
        axs[0, 1].set_xlabel('Timestep')
        axs[0, 1].set_ylabel('Angular Velocity (rad/s)')
        axs[0, 1].legend()
        axs[0, 1].grid(True)

        axs[0, 2].plot(velocity_history[:,3], label='u_1')
        axs[0, 2].plot(velocity_history[:,4], label='u_2')
        axs[0, 2].set_title('u1 and u2 History')
        axs[0, 2].set_xlabel('Timestep')
        axs[0, 2].set_ylabel('Velocity (m/s)')
        axs[0, 2].legend()
        axs[0, 2].grid(True)

        # 3) x_history and y_history (trajectory)
        axs[1, 0].plot(state_history[:,0], state_history[:,1], marker='o', label='Trajectory')
        axs[1, 0].set_title('Robot Trajectory (x vs y)')
        axs[1, 0].set_xlabel('x (m)')
        axs[1, 0].set_ylabel('y (m)')
        axs[1, 0].axis('equal')
        axs[1, 0].grid(True)
        axs[1, 0].legend()

        # 4) th_history
        axs[1, 1].plot(state_history[:,2], label='theta', color='orange')
        axs[1, 1].set_title('Theta (Orientation) History')
        axs[1, 1].set_xlabel('Timestep')
        axs[1, 1].set_ylabel('Theta (rad)')
        axs[1, 1].legend()
        axs[1, 1].grid(True)

        axs[1, 2].plot(state_history[:,3], label='k1')
        axs[1, 2].plot(state_history[:,4], label='k2')
        axs[1, 2].set_title('Curvature history')
        axs[1, 2].set_xlabel('Timestep')
        axs[1, 2].set_ylabel('k (m^-1)')
        axs[1, 2].axis('equal')
        axs[1, 2].grid(True)
        axs[1, 2].legend()

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    STIFFNESS_SIMULATION = {
        1:  0.5,  
        -1: -0.5, 
        0:  0.0   
    }
    
    robot = robot_state.Model(1, 0, 0, 0)

    robot.t1 = 22
    robot.t2 = 22

    # targets = [[0.5, 0.5, np.pi/4, 0, 0],
    #            [0.12, 0.7, 2*np.pi/3, 15, 0]]

    targets = [
               [0.12, 0.5, 2*np.pi/3, 15, 0]]
    
    mm_controller = mm.MotionMorphologyControl(robot)

    count = 0

    for target in targets:
        print(f'Target: {target}')
        target_robot = robot_state.Model(2, *target)

        state_history = [robot.config]
        stiffness_history = [robot.stiffness]
        velocity_history = [[0.0] * 5]
        stiffness_actions_history = [[0, 0]]

        is_finished = False

        while not is_finished:
            print()
            current_vel, stiff_transitions, q_new, is_finished = mm_controller.go_to_target(target)
            
            robot.config = q_new

            robot.t1 += STIFFNESS_SIMULATION.get(stiff_transitions[0], 0.0)
            robot.t2 += STIFFNESS_SIMULATION.get(stiff_transitions[1], 0.0)

            print(f"Velocity: {current_vel}")

            state_history.append(robot.config)
            stiffness_history.append(robot.stiffness)
            velocity_history.append(current_vel)
            stiffness_actions_history.append(stiff_transitions)

            count += 1

    # Create the animation engine with the chosen scenario
    animator = RobotAnimation(target_robot, state_history, stiffness_history, 
                              velocity_history, stiffness_actions_history)

    # Run the process
    # animator.run_and_save()

    animator.plot_data()

    