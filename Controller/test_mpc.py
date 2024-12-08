import env
import func
import robot2sr_controller as rsr_ctrl
import time
import matplotlib.pyplot as plt
import numpy as np

agent_controller = rsr_ctrl.Controller()

# simulation = True
simulation = False

def get_displacement():
    while True:
        try:
            # Get input from user
            user_input = input("Enter desired displacement (delta_x, delta_y, delta_theta) or 'q' to quit: ")
            
            # Check for quit command
            if user_input.lower() == 'q':
                print("Exiting program...")
                return True, 0, 0, 0
            
            # Split the input string by comma and convert to float
            delta_x, delta_y, delta_theta = map(float, user_input.split(','))
            
            return False, delta_x, delta_y, delta_theta
            
        except ValueError:
            print("Invalid input. Please enter three numbers separated by commas or 'q' to quit.")
        except Exception as e:
            print(f"Error: {e}. Please try again.")

def moveRobot(agent, target, rgb_camera, start_time):
    v_r = [0.0] * 3
    v_s = [0.0] * 2

    target_s = [0, 0]

    timestamps = []
    robot_states = []
    target_vel = []

    finish = False

    while True:
        elapsed_time = time.perf_counter() - start_time
            
        timestamps.append(elapsed_time)
        robot_states.append(agent.pose)
        
        if func.close2Goal(agent.pose, target[:3]) or rgb_camera.finish:
            v_r = [0.0] * 3
            finish = True
        else:
            v_r, q_new = agent_controller.mpcRM(agent, target[:3], v_r)

        v = v_r + v_s
        print(f'Stiffness: {target_s}')
        print(f'Velocity: {v}')

        target_vel.append(v_r)

        agent_controller.move(agent, v, target_s)

        if simulation:
            agent.config = q_new

        if finish:
            break

    elapsed_time = time.perf_counter() - start_time
            
    timestamps.append(elapsed_time)
    robot_states.append(agent.pose)
    target_vel.append(v_r)

    tracking_data = {
        'time': timestamps,
        'robot_states': robot_states,
        'target_vel': target_vel,
    }

    return tracking_data

def plot_data(all_data):
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle('Tracking Performance')

    # Colors for different velocities
    vel_colors = ['r', 'g', 'b']
    vel_labels = ['v_x', 'v_y', 'ω']

    all_x_errors = []
    all_y_errors = []
    all_theta_errors = []
    all_vel = []

    # Process data for each target
    for i, data in enumerate(all_data):
        target = np.array(data['target'])  # [x, y, theta]
        time = np.array(data['tracking']['time'])
        robot_states = np.array(data['tracking']['robot_states'])  # measured poses
        target_vel = data['tracking']['target_vel']  # velocities

        # Calculate errors (with signs)
        x_error = target[0] - robot_states[:, 0]
        y_error = target[1] - robot_states[:, 1]
        theta_error = target[2] - robot_states[:, 2]
        # Normalize theta error to [-pi, pi]
        theta_error = np.arctan2(np.sin(theta_error), np.cos(theta_error))

        all_x_errors.extend(x_error)
        all_y_errors.extend(y_error)
        all_theta_errors.extend(theta_error)
        all_vel.extend(target_vel)

    all_vel = np.array(all_vel)

    # Plot X error and velocities
    ax1.plot(all_x_errors, 'k-', alpha=0.5, label=f'Error (Target {i+1})')
    for j in range(3):
        ax1.plot(all_vel[:, j], vel_colors[j], 
                linestyle='--', alpha=0.5, 
                label=f'{vel_labels[j]} (Target {i+1})')

    # Plot Y error and velocities
    ax2.plot(all_y_errors, 'k-', alpha=0.5, label=f'Error (Target {i+1})')
    for j in range(3):
        ax2.plot(all_vel[:, j], vel_colors[j], 
                linestyle='--', alpha=0.5, 
                label=f'{vel_labels[j]} (Target {i+1})')

    # Plot Theta error and velocities
    ax3.plot(all_theta_errors, 'k-', alpha=0.5, label=f'Error (Target {i+1})')
    for j in range(3):
        ax3.plot(all_vel[:, j], vel_colors[j], 
                linestyle='--', alpha=0.5, 
                label=f'{vel_labels[j]} (Target {i+1})')

    # Customize plots
    ax1.set_ylabel('X Error (m) / Velocities')
    ax1.grid(True)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    ax2.set_ylabel('Y Error (m) / Velocities')
    ax2.grid(True)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    ax3.set_ylabel('Theta Error (rad) / Velocities')
    ax3.set_xlabel('Time (s)')
    ax3.grid(True)
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Adjust layout to prevent label overlap
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Set up the environment 
    env_observer = env.Observer(simulation)
    env_observer.run()

    all_data = []
    start_time = time.perf_counter()

    while True:
        exit_status, delta_x, delta_y, delta_theta = get_displacement()

        if exit_status or env_observer.rgb_camera.finish:
            env_observer.rgb_camera.finish = True
            break

        target_pose = [env_observer.agent.x + delta_x, 
                        env_observer.agent.y + delta_y,
                        env_observer.agent.theta + delta_theta]
        
        target_q = target_pose + env_observer.agent.curvature
        env_observer.rgb_camera.target_robot_config = target_q

        tracking_data = moveRobot(env_observer.agent, target_q, env_observer.rgb_camera, start_time)

        all_data.append({
            'target': target_pose,
            'tracking': tracking_data
        })

    plot_data(all_data)
