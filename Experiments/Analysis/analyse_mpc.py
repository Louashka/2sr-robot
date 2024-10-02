
import json
import numpy as np
import matplotlib.pyplot as plt

def parse_data(data: dict) -> tuple:
    target_pos = np.array([[data['target pos']['x'], data['target pos']['y']]]).T
    target_lu1 = np.array([[data['target lu1']['x'], data['target lu1']['y']]]).T
    target_lu2 = np.array([[data['target lu2']['x'], data['target lu2']['y']]]).T

    pos = np.array([data['tracking']['pos']['x'], data['tracking']['pos']['y']])
    lu1 = np.array([data['tracking']['lu1']['x'], data['tracking']['lu1']['y']])
    lu2 = np.array([data['tracking']['lu2']['x'], data['tracking']['lu2']['y']])

    return target_pos, target_lu1, target_lu2, pos, lu1, lu2


if __name__ == "__main__":
    json_file_name = 'D:/Robot 2SR/2sr-swarm-control/Experiments/Data/Tracking/SM2/sm2_data_2024-09-24_14-57-16.json'
    json_file = open(json_file_name)
    data = json.load(json_file)

    target_pos, target_lu1, target_lu2, pos, lu1, lu2 = parse_data(data)

    # Calculate errors
    pos_error = np.linalg.norm(target_pos - pos, axis=0)
    lu1_error = np.linalg.norm(target_lu1 - lu1, axis=0)
    lu2_error = np.linalg.norm(target_lu2 - lu2, axis=0)

    total_error = pos_error + lu1_error + lu2_error

    # Create a new figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the errors
    ax.plot(pos_error, label='Position Error')
    ax.plot(lu1_error, label='LU1 Error')
    ax.plot(lu2_error, label='LU2 Error')
    ax.plot(total_error, label='Total Error')

    # Set labels and title
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Error (m)')
    ax.set_title('Tracking Errors over Time')

    # Add legend
    ax.legend()

    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)

    # Show the plot
    plt.tight_layout()
    plt.show()

    

