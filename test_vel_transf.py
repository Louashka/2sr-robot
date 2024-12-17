import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from View import plotlib

# Initialize circle parameters
radius = 0.5
q_b = np.array([0.0, 0.0, np.pi/4])

# Initialize velocities (vx, vy, omega)
# v_b = np.array([np.sqrt(2)/10, np.sqrt(2)/10, -0.1])
v_b = np.array([0, 0, 0.1])
# v_b = np.array([0.0, 0.2, 0.5])

# Time step and total time
dt = 0.1
total_time = 10

# Lists to store circle's position and orientation over time
x_history = [q_b[0]]
y_history = [q_b[1]]
theta_history = [q_b[2]]

delta_bs = 11 * np.pi/12
d_bs = [radius * np.cos(delta_bs), radius * np.sin(delta_bs), delta_bs + np.pi/2]
rot_b = np.array([[np.cos(q_b[2]), -np.sin(q_b[2]), 0],
                  [np.sin(q_b[2]), np.cos(q_b[2]), 0],
                  [0, 0, 1]])
q_s = q_b + rot_b.dot(d_bs)

q_r = [q_b[0], q_b[1]-radius, 0.0, 1/radius, 1/radius]


# Function to update the circle's position and orientation
def update(frame):
    global q_b, q_s

    rot_b = np.array([[np.cos(q_b[2]), -np.sin(q_b[2]), 0],
                      [np.sin(q_b[2]), np.cos(q_b[2]), 0],
                      [0, 0, 1]])
    
    # Update position and orientation
    q_b_dot = rot_b.dot(v_b)
    q_b += q_b_dot * dt

    # Store new position and orientation
    x_history.append(q_b[0])
    y_history.append(q_b[1])
    theta_history.append(q_b[2])

    th_rs = q_s[2] - q_r[2]
    dist_rs = (q_s[:2] - q_r[:2]).reshape(2,1)

    rot_bs = np.array([[np.cos(delta_bs + np.pi/2), -np.sin(delta_bs + np.pi/2)],
                       [np.sin(delta_bs + np.pi/2), np.cos(delta_bs + np.pi/2)]])
    Ad_bs_inv = np.block([[rot_bs.T, rot_bs.T.dot(np.array([[-d_bs[1]], [d_bs[0]]]))],
                          [np.zeros((1, 2)), 1]])
    
    v_s = Ad_bs_inv.dot(v_b)

    rot_s = np.array([[np.cos(q_s[2]), -np.sin(q_s[2]), 0],
                      [np.sin(q_s[2]), np.cos(q_s[2]), 0],
                      [0, 0, 1]])
    
    q_s_dot = rot_s.dot(v_s)
    q_s += q_s_dot * dt


    # -------------------------------------------------
    arc1_x, arc1_y, arc1_th = plotlib.arc(q_r)
    arc2_x, arc2_y, arc2_th = plotlib.arc(q_r, 2)

    rot_rs = np.array([[np.cos(th_rs), -np.sin(th_rs)],
                       [np.sin(th_rs), np.cos(th_rs)]]) 
    
    rot_rw = np.array([[np.cos(-q_r[2]), -np.sin(-q_r[2])],
                       [np.sin(-q_r[2]), np.cos(-q_r[2])]])
    
    pos_rs = rot_rw.dot(dist_rs)
    
    Ad_rs = np.block([[rot_rs, np.array([[pos_rs[1,0]], [-pos_rs[0,0]]])],
                      [np.zeros((1, 2)), 1]])
    
    v_r = Ad_rs.dot(v_s)

    # Ad_rs_inv_T = np.block([[rot_rs, np.zeros((2, 1))], 
    #                         [np.array([[-dist_rs[1], dist_rs[0]]]).dot(rot_rs), 1]])
    
    # v_r = Ad_rs_inv_T.dot(v_s)

    rot_r = np.array([[np.cos(q_r[2]), -np.sin(q_r[2]), 0],
                      [np.sin(q_r[2]), np.cos(q_r[2]), 0],
                      [0, 0, 1]])
    
    q_r_dot = rot_r.dot(v_r)
    q_r[:3] += q_r_dot * dt

    # -------------------------------------------------

    # Clear the plot and redraw
    plt.clf()
    arrow_length = 0.1

    # -------------------------------------------------
    plt.plot(arc1_x, arc1_y, 'g-')
    plt.plot(arc2_x, arc2_y, 'g-')
    plt.plot(q_r[0], q_r[1], 'r*')

    plt.plot(arc1_x[-1], arc1_y[-1], 'r*')
    arrow_x = [arc1_x[-1], arc1_x[-1] + arrow_length * np.cos(arc1_th)]
    arrow_y = [arc1_y[-1], arc1_y[-1] + arrow_length * np.sin(arc1_th)]
    plt.arrow(arrow_x[0], arrow_y[0], arrow_x[1] - arrow_x[0], arrow_y[1] - arrow_y[0], 
              head_width=0.03, head_length=0.05, fc='r', ec='r')
    # -------------------------------------------------
    
    
    # Plot the circle
    circle = plt.Circle((q_b[0], q_b[1]), radius, fill=False)
    plt.gca().add_artist(circle)
    
    # Plot the line indicating orientation
    arrow_x = [q_b[0], q_b[0] + arrow_length * np.cos(q_b[2])]
    arrow_y = [q_b[1], q_b[1] + arrow_length * np.sin(q_b[2])]
    plt.arrow(arrow_x[0], arrow_y[0], arrow_x[1] - arrow_x[0], arrow_y[1] - arrow_y[0], 
              head_width=0.03, head_length=0.05, fc='r', ec='r')
    
    plt.plot(q_s[0], q_s[1], 'b*')
    arrow_x = [q_s[0], q_s[0] + arrow_length * np.cos(q_s[2])]
    arrow_y = [q_s[1], q_s[1] + arrow_length * np.sin(q_s[2])]
    plt.arrow(arrow_x[0], arrow_y[0], arrow_x[1] - arrow_x[0], arrow_y[1] - arrow_y[0], 
              head_width=0.03, head_length=0.05, fc='r', ec='r')
    
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
anim.save('vel_trans.mp4', writer='ffmpeg', fps=16)  

plt.show()
