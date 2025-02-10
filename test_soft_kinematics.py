import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from Model import global_var as gv, splines
from typing import List

def arc(q, seg=1) -> list:
    s = np.linspace(0, gv.L_VSS, 50)

    flag = -1 if seg == 1 else 1

    gamma_array = q[2] + flag * q[2 + seg] * s

    x_0 = q[0]
    y_0 = q[1]

    if q[2 + seg] == 0:
        x = x_0 + [0, flag * gv.L_VSS * np.cos(q[2])]
        y = y_0 + [0, flag * gv.L_VSS * np.sin(q[2])]
    else:
        x = x_0 + np.sin(gamma_array) / \
            q[2 + seg] - np.sin(q[2]) / q[2 + seg]
        y = y_0 - np.cos(gamma_array) / \
            q[2 + seg] + np.cos(q[2]) / q[2 + seg]
        
    return [x, y]


def get_jacobian(q:np.ndarray, s: List[float]) -> np.ndarray:
    cardioid1 = splines.Cardioid(1)
    cardioid2 = splines.Cardioid(2)
    cardioid3 = splines.Cardioid(3)

    if all(s):
        spiral1 = spiral2 = cardioid3
    else:
        spiral1 = cardioid1
        spiral2 = cardioid2

    # k1_ratio = spiral2.k_dot(q[4]) / cardioid1.k_dot(q[4])
    # k2_ratio = spiral2.k_dot(q[3]) / cardioid1.k_dot(q[3])

    k1_ratio = cardioid1.delta_phi / (spiral2.delta_phi * spiral2.rho(q[4]))
    k2_ratio = cardioid1.delta_phi / (spiral2.delta_phi * spiral2.rho(q[3]))

    pos_lu1 = cardioid1.pos_dot(q[2], q[4], 2, 1)
    pos_lu2 = cardioid1.pos_dot(q[2], q[3], 1, 2)

    J = np.array([[k1_ratio * pos_lu1[0], k2_ratio * pos_lu2[0]],
                  [k1_ratio * pos_lu1[1], k2_ratio * pos_lu2[1]],
                  [spiral2.th_dot(q[4]), spiral2.th_dot(q[3])],
                  [-spiral1.k_dot(q[3]), spiral2.k_dot(q[3])],
                  [-spiral2.k_dot(q[4]), spiral1.k_dot(q[4])]])
    
    stiffness_array = np.array([[s[1], s[0]],
                                [s[1], s[0]],
                                [s[1], s[0]],
                                [s[0], s[0]],
                                [s[1], s[1]]])
    
    return np.multiply(stiffness_array, J)


fig, ax = plt.subplots()

dt = 0.1  # step size

q = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
s = [1, 1]

v = np.array([0.0, 0.05])

arc1, = ax.plot([], [], lw=3, color="blue")
arc2, = ax.plot([], [], lw=3, color="blue")
centre, = ax.plot([], [], lw=5, marker="o", color="black")
end_points, = ax.plot([], [], 'ro', markersize=2)  # New line for end points

end_point_history = []  # List to store end point coordinates
q_dot_history = []

def init():
    global ax

    ax.set_xlim([-0.15, 0.15])
    ax.set_ylim([-0.15, 0.15])
    ax.set_aspect("equal")

def update(i):
    global q, arc1, arc2, centre, end_points, end_point_history, q_dot_history

    # delta = np.array([
    #                     int(s[0] and not s[1]),    # delta1
    #                     int(not s[0] and s[1]),    # delta2
    #                     int(all(s))                # delta3
    #                 ])
    
    # J_array = np.array([get_jacobian(i, q, s) for i in range(1, 4)])
    
    # # Perform matrix multiplication
    # # J_array shape: (3, m, n), delta shape: (3,)
    # # Result shape: (m, n)
    # J =  np.tensordot(J_array, delta, axes=([0], [0]))

    J = get_jacobian(q, s)
    
    q_dot = J@v
    q_dot_history.append(q_dot.tolist())
    q += q_dot * dt

    q[3] = round(q[3], 4)
    q[4] = round(q[4], 4)

    centre.set_data([q[0]], [q[1]])

    seg1 = arc(q, 1)
    seg2 = arc(q, 2)

    arc1.set_data(seg1[0], seg1[1])
    arc2.set_data(seg2[0], seg2[1])

    # Store the end point of seg2
    end_point_history.append((seg1[0][-1], seg1[1][0-1]))

    # Update the end_points plot with all historical points
    x_history, y_history = zip(*end_point_history)
    end_points.set_data(x_history, y_history)

    if s[0] == 0:
        arc1.set_color("blue")
    else:
        arc1.set_color("red")

    if s[1] == 0:
        arc2.set_color("blue")
    else:
        arc2.set_color("red")

    return arc1, arc2, centre, end_points

# def arc_new(theta0, k, seg=1):
#     l = np.linspace(0, gv.L_VSS, 50)
#     flag = -1 if seg == 1 else 1
#     theta_array = theta0 + flag * k * l

#     if k == 0:
#         x = np.array([0, flag * gv.L_VSS * np.cos(theta0)])
#         y = np.array([0, flag * gv.L_VSS * np.sin(theta0)])
#     else:
#         x = np.sin(theta_array) / k - np.sin(theta0) / k
#         y = -np.cos(theta_array) / k + np.cos(theta0) / k

#     theta = theta_array[-1]
        
#     return [x, y, theta % (2 * np.pi)]

if __name__ == "__main__":

    anim = FuncAnimation(fig, update, frames=30, init_func=init, interval=100, repeat=False)

    plt.show()

    # q_dot_history = np.array(q_dot_history)

    # fig, axs = plt.subplots(2, 2, figsize=(16, 8))

    # axs[0,0].plot(1000 * q_dot_history[:,0], 'k-')
    # axs[0,1].plot(1000 * q_dot_history[:,1], 'k-')
    # axs[1,0].plot(q_dot_history[:,3], 'k-')
    # axs[1,1].plot(q_dot_history[:,4], 'k-')

    # for i in range(2):
    #     for j in range(2):
    #         axs[i,j].axis('equal')

    # plt.show()

