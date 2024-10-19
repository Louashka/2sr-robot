from Model import global_var as gv
import numpy as np
import matplotlib.pyplot as plt

def arc(q_0, k, seg=1):
    l = np.linspace(0, gv.L_VSS, 50)
    flag = -1 if seg == 1 else 1
    
    theta_array = q_0[2,0] + flag * k * l

    if k == 0:
        x = np.array([0, flag * gv.L_VSS * np.cos(q_0[2,0])])
        y = np.array([0, flag * gv.L_VSS * np.sin(q_0[2,0])])
    else:
        x = np.sin(theta_array) / k - np.sin(q_0[2]) / k
        y = -np.cos(theta_array) / k + np.cos(q_0[2]) / k

    x += q_0[0,0]
    y += q_0[1,0]
    theta = theta_array[-1]
        
    return [x, y, theta % (2 * np.pi)]

rb = np.array([[0.0, 0.0, 0.0]]).T
k = np.array([-38, 38])

seg1 = arc(rb, k[0])
seg2 = arc(rb, k[1], 2)
cp1 = np.array([[seg1[0][-1], seg1[1][-1], seg1[2]]]).T
cp3 = np.array([[seg2[0][-1], seg2[1][-1], seg2[2]]]).T

rb_dot = np.array([[-0.047, 0.061, 0.1]]).T

t = 0

while t < 500:
    rb += rb_dot * gv.DT

    seg1 = arc(rb, k[0])
    seg2 = arc(rb, k[1], 2)

    # Calculate Euclidean distances
    l_s1 = np.sqrt((rb[0,0] - seg1[0][-1])**2 + (rb[1,0] - seg1[1][-1])**2)
    l_s3 = np.sqrt((rb[0,0] - seg2[0][-1])**2 + (rb[1,0] - seg2[1][-1])**2)

    # print(f"Distance from robot to end of segment 1: {dist_rb_seg1:.4f}")
    # print(f"Distance from robot to end of segment 2: {dist_rb_seg2:.4f}")
    
    T1 = np.array([[1, 0, l_s1 * np.sin(rb[2,0] - gv.L_VSS * k[0] / 2)], 
                   [0, 1, -l_s1 * np.cos(rb[2,0] - gv.L_VSS * k[0] / 2)], 
                   [0, 0, 1]])
    
    T3 = np.array([[1, 0, -l_s3 * np.sin(rb[2,0] + gv.L_VSS * k[1] / 2)], 
                   [0, 1,  l_s3 * np.cos(rb[2,0] + gv.L_VSS * k[1] / 2)], 
                   [0, 0, 1]])
    
    cp1_dot = T1.dot(rb_dot)
    cp3_dot = T3.dot(rb_dot)

    cp1 += cp1_dot * gv.DT
    cp3 += cp3_dot * gv.DT

    t += 1

    # Clear the previous plot
    plt.clf()

    # Plot seg1 as a curved line
    plt.plot(seg1[0], seg1[1], 'b-', linewidth=2, label='Segment 1')

    # Plot seg2 as a curved line
    plt.plot(seg2[0], seg2[1], 'b-', linewidth=2, label='Segment 2')

    # Plot rb as a dot
    plt.plot(rb[0,0], rb[1,0], 'ro', markersize=5, label='Robot Center')
    plt.plot(cp1[0,0], cp1[1,0], 'ro', markersize=5, label='CP 1')
    plt.plot(cp3[0,0], cp3[1,0], 'ro', markersize=5, label='CP 3')

    # Set equal aspect ratio
    plt.axis('equal')

    # Add labels and title
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Robot Configuration at t = {t:.2f}')

    # Add legend
    plt.legend()

    # Display the plot
    plt.draw()
    plt.pause(0.01)  # Pause to allow the plot to update

# Move plt.show() outside the while loop to keep the window open
plt.show()
