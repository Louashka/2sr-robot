from gekko import GEKKO
import numpy as np


# --- Example Usage ---
if __name__ == '__main__':
    # 2. Define our start and end points
    robot_current_pose = [0, 0, 0]
    robot_target_pose = [5, 5, np.pi/2]
    
    print(f"Starting at {robot_current_pose}")
    print(f"Going to {robot_target_pose}\n")

    vx, vy, w = 0.0, 0.0, 0.0

    # 1. Setup the MPC controller once
    m = GEKKO(remote=False)
    m.time = np.linspace(0, 0.2, 21)

    # --- Manipulated Variables (MVs) ---
    # The velocities in the robot's own frame.
    m.v_x = m.MV(value=vx, lb=-0.5, ub=0.5)
    m.v_y = m.MV(value=vy, lb=-0.3, ub=0.3)
    m.omega = m.MV(value=w, lb=-np.pi/2, ub=np.pi/2)

    # Turn on the MVs for the optimizer to use
    m.v_x.STATUS = 1
    m.v_y.STATUS = 1
    m.omega.STATUS = 1

    # Add cost to changes to encourage smooth control actions
    m.v_x.DCOST = 1
    m.v_y.DCOST = 1
    m.omega.DCOST = 0.5 # Penalize turning changes less/more depending on desired behavior

    # --- Controlled Variables (CVs) / State Variables (SVs) ---
    # The robot's pose in the world frame.
    m.x = m.CV()
    m.y = m.CV()
    m.theta = m.CV()

    m.x.FSTATUS = 1
    m.y.FSTATUS = 1
    m.theta.FSTATUS = 1

    # Turn on the CVs for control and define their setpoints
    m.x.STATUS = 1; m.y.STATUS = 1; m.theta.STATUS = 1
    m.options.CV_TYPE = 2  # Use squared error for the objective function
    m.x.SP = robot_target_pose[0]
    m.y.SP = robot_target_pose[1]
    m.theta.SP = robot_target_pose[2]

    # Set a reference trajectory for the setpoints to follow (smooth approach)
    m.x.TR_INIT = 1; m.y.TR_INIT = 1; m.theta.TR_INIT = 1
    m.x.TAU = 1.0; m.y.TAU = 1.0; m.theta.TAU = 1.2 # Time constant for trajectory

    # --- The Kinematic Model ---
    # These equations define how the MVs affect the CVs. This is the "physics".
    # It converts robot-frame velocities to world-frame pose changes.
    m.Equation(m.x.dt() == m.cos(m.theta) * m.v_x - m.sin(m.theta) * m.v_y)
    m.Equation(m.y.dt() == m.sin(m.theta) * m.v_x + m.cos(m.theta) * m.v_y)
    m.Equation(m.theta.dt() == m.omega)
    
    # --- Solver Settings ---
    m.options.IMODE = 6  # MPC Control Mode
    m.options.SOLVER = 3 # IPOPT solver is good for nonlinear problems

    # 3. In a real robot, this would be your main control loop (e.g., running at 10 Hz)
    for i in range(50): # Simulate 50 steps
        print(f"--- Step {i+1} ---")
        print(f"Current Pose: [{robot_current_pose[0]:.2f}, {robot_current_pose[1]:.2f}, {robot_current_pose[2]:.2f}]")
        
        m.x.MEAS = robot_current_pose[0]
        m.y.MEAS = robot_current_pose[1]
        m.theta.MEAS = robot_current_pose[2]

        # Solve for the optimal control action
        m.solve(disp=False)
        
        # Get the first (current) optimal control action from the solution
        optimal_velocities = [m.v_x.NEWVAL, m.v_y.NEWVAL, m.omega.NEWVAL]

        print(f"Optimal Velocities: [vx={optimal_velocities[0]:.2f}, vy={optimal_velocities[1]:.2f}, w={optimal_velocities[2]:.2f}]")

        # --- Simulate the robot moving for one time step (e.g., dt=0.1s) ---
        dt = 0.1
        vx, vy, w = optimal_velocities

        m.v_x.VALUE = vx
        m.v_y.VALUE = vy
        m.omega.VALUE = w
        
        robot_current_pose[0] += (np.cos(robot_current_pose[-1]) * vx - np.sin(robot_current_pose[-1]) * vy) * dt
        robot_current_pose[1] += (np.sin(robot_current_pose[-1]) * vx + np.cos(robot_current_pose[-1]) * vy) * dt
        robot_current_pose[2] += w * dt
        
        # Check for arrival
        dist_to_target = np.linalg.norm(np.array(robot_current_pose[:2]) - np.array(robot_target_pose[:2]))
        if dist_to_target < 0.1:
            print("\nArrived at target!")
            break