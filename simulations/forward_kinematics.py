import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from kinematics import HybridKinematics
from plotlib import RobotPlot
from entities import robot_state

def main():
    """
    Main function to set up and run the robot simulation and animation.
    """
    # --- 1. Simulation Setup ---
    kinematics_engine = HybridKinematics()
    initial_state = [0.0] * 5
    robot = robot_state.Model(1, *initial_state) # Start facing up
    dt = 0.1  # Time step for integration [s]

    # --- 2. Define a Control Sequence ---
    # This defines a series of actions for the robot to perform.
    # Each phase has a duration, a stiffness configuration, and a control input vector.
    # u = [vx, vy, omega, v1, v2]
    # s = [s1, s2] (0=rigid, 1=flexible)
    CONTROL_PHASES = [
        {"duration": 1.0, "s": [0, 0], "temp": [22.0, 22.0], "u": [0, 0, 0, 0, 0], "desc": "Idle"},
        {"duration": 2.0, "s": [1, 0], "temp": [64.0, 22.0], "u": [0, 0, 0, 0, 0.05], "desc": "Flex segment 1"},
        {"duration": 2.0, "s": [1, 0], "temp": [64.0, 22.0], "u": [0, 0, 0, 0, -0.05], "desc": "Unflex segment 1"},
        {"duration": 2.0, "s": [0, 1], "temp": [22.0, 64.0], "u": [0, 0, 0, -0.05, 0], "desc": "Flex segment 2"},
        {"duration": 2.0, "s": [0, 1], "temp": [22.0, 64.0], "u": [0, 0, 0, 0.05, 0], "desc": "Unflex segment 2"},
        {"duration": 3.0, "s": [0, 0], "temp": [22.0, 22.0], "u": [0, 0.1, 0, 0, 0], "desc": "Move up (rigid)"},
        {"duration": 2.0, "s": [1, 1], "temp": [64.0, 64.0], "u": [0, 0, 0, 0.06, 0.06], "desc": "Rotate (flexible)"},
        {"duration": 2.0, "s": [1, 1], "temp": [64.0, 64.0], "u": [0, 0, 0, -0.045, 0.045], "desc": "Flex both segments"},
        {"duration": 3.0, "s": [1, 1], "temp": [64.0, 64.0], "u": [0, 0, 0, 0.077, 0.0175], "desc": "Unflex both segments"}
    ]

    # Calculate total frames and create a time-to-phase mapping
    total_time = sum(p['duration'] for p in CONTROL_PHASES)
    num_frames = int(total_time / dt)
    
    phase_timeline = []
    current_time = 0
    for i, phase in enumerate(CONTROL_PHASES):
        start_frame = int(current_time / dt)
        current_time += phase['duration']
        end_frame = int(current_time / dt)
        for _ in range(start_frame, end_frame):
            phase_timeline.append(i)
    phase_timeline.append(len(CONTROL_PHASES) - 1) 

    # --- 3. Plotting Setup ---
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal', 'box')
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.3, 0.7)
    ax.set_title("2SR Robot Forward Kinematics Simulation")
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Instantiate the plotter from plotlib.py
    robot_plotter = RobotPlot(ax)

    # --- 4. Animation Core Logic ---
    def update(frame: int):
        """
        Called for each frame of the animation.
        It performs one step of the forward kinematics.
        """
        nonlocal robot
        
        # a. Determine the current control inputs from the sequence
        phase_index = phase_timeline[frame]
        current_phase = CONTROL_PHASES[phase_index]
        s_flags = current_phase["s"]
        u_control = np.array(current_phase["u"])
        
        # b. Get the unified Jacobian for the current state
        J = kinematics_engine.get_unified_jacobian(robot, s_flags)
        
        # c. Calculate the change in state (q_dot)
        q_dot = J @ u_control
        
        # d. Integrate to get the new state (Euler method)
        robot.config += q_dot * dt
        robot.stiffness = current_phase['s']
        robot.temp = current_phase['temp']
        
        # e. Update the plot with the new robot state
        artists = robot_plotter.plot_robot(robot)
        
        # Update title 
        ax.set_title(f"2SR Robot Simulation | Step: {current_phase['desc']}")
        
        return artists

    # --- 5. Create and Run the Animation ---
    init_state_artists = robot_plotter.plot_robot(robot)
    
    ani = animation.FuncAnimation(
        fig=fig,
        func=update,
        frames=num_frames,
        interval=dt * 1000, 
        repeat=True
    )
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()