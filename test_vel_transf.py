import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Initialize circle parameters
radius = 0.5
q = np.array([0.0, 0.0, 0.0])

# Initialize velocities (vx, vy, omega)
v_b = np.array([0.1, 0.2, 0.3])
# v_b = np.array([0.0, 0.2, 0.5])

# Time step and total time
dt = 0.1
total_time = 10

# Lists to store circle's position and orientation over time
x_history = [q[0]]
y_history = [q[1]]
theta_history = [q[2]]

tracking_theta = np.pi/4
tracking_point_b = np.array([radius * np.cos(tracking_theta),
                             radius * np.sin(tracking_theta),
                             tracking_theta])
tracking_point_s = np.array([radius * np.cos(tracking_theta),
                             radius * np.sin(tracking_theta),
                             tracking_theta])

# Function to update the circle's position and orientation
def update(frame):
    global q, tracking_point_b, tracking_point_s

    rot_sb = np.array([[np.cos(q[2]), -np.sin(q[2]), 0],
                           [np.sin(q[2]), np.cos(q[2]), 0],
                           [0, 0, 1]])
    
    # Update position and orientation
    q_dot_circle = rot_sb.dot(v_b)
    q += q_dot_circle * dt

    rot_bs = np.array([[np.cos(tracking_theta), -np.sin(tracking_theta)],
                       [np.sin(tracking_theta), np.cos(tracking_theta)]])
    Ad_bs_inv = np.block([[rot_bs.T, rot_bs.T.dot(np.array([[-tracking_point_b[1]], [tracking_point_b[0]]]))],
                          [np.zeros((1, 2)), 1]])
    
    v_s = Ad_bs_inv.dot(v_b)

    rot_s = np.array([[np.cos(q[2] + tracking_theta), -np.sin(q[2] + tracking_theta), 0],
                      [np.sin(q[2] + tracking_theta), np.cos(q[2] + tracking_theta), 0],
                      [0, 0, 1]])
    
    tracking_point_dot = rot_s.dot(v_s)
    tracking_point_s += tracking_point_dot * dt
    
    # Store new position and orientation
    x_history.append(q[0])
    y_history.append(q[1])
    theta_history.append(q[2])
    
    # Clear the plot and redraw
    plt.clf()
    
    # Plot the circle
    circle = plt.Circle((q[0], q[1]), radius, fill=False)
    plt.gca().add_artist(circle)
    
    # Plot the line indicating orientation
    line_x = [q[0], q[0] + radius * np.cos(q[2])]
    line_y = [q[1], q[1] + radius * np.sin(q[2])]
    plt.plot(line_x, line_y, 'r-')
    
    plt.plot(tracking_point_s[0], tracking_point_s[1], 'r*')
    
    # Plot the trajectory
    plt.plot(x_history, y_history, 'b--', alpha=0.5)
    
    plt.xlim(min(x_history) - radius - 0.5, max(x_history) + radius + 0.5)
    plt.ylim(min(y_history) - radius - 0.5, max(y_history) + radius + 0.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f'Time: {frame*dt:.1f}s')
    plt.grid(True)

# Create the animation
fig = plt.figure(figsize=(8, 8))
anim = FuncAnimation(fig, update, frames=int(total_time/dt), interval=50, repeat=False)

plt.show()
