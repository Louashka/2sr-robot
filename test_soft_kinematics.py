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


def get_jacobian(n: int, q:np.ndarray, s: List[float]) -> np.ndarray:
    if n == 3:
        spiral1 = spiral2 = splines.LogSpiral(3)
    else:
        spiral1 = splines.LogSpiral(1)
        spiral2 = splines.LogSpiral(2)

    pos_lu1 = spiral2.get_pos_dot(q[2], q[4], 2, 1)
    pos_lu2 = spiral2.get_pos_dot(q[2], q[3], 1, 2)

    J = np.array([[pos_lu1[0], pos_lu2[0]],
                  [pos_lu1[1], pos_lu2[1]],
                  [spiral2.get_th_dot(q[4]), spiral2.get_th_dot(q[3])],
                  [-spiral1.get_k_dot(q[3]), spiral2.get_k_dot(q[3])],
                  [-spiral2.get_k_dot(q[4]), spiral1.get_k_dot(q[4])]])
    
    stiffness_array = np.array([[s[1], s[0]],
                                [s[1], s[0]],
                                [s[1], s[0]],
                                [s[0], s[0]],
                                [s[1], s[1]]])
    
    return np.multiply(stiffness_array, J)


fig, ax = plt.subplots()

dt = 0.1  # step size

q = np.array([0.0, 0.0, np.pi/5, 0.0, 0.0])
s = [1, 1]

v = np.array([-0.03, 0.0])

arc1, = ax.plot([], [], lw=3, color="blue")
arc2, = ax.plot([], [], lw=3, color="blue")
centre, = ax.plot([], [], lw=5, marker="o", color="black")

def init():
    global ax

    ax.set_xlim([-0.25, 0.25])
    ax.set_ylim([-0.25, 0.25])
    ax.set_aspect("equal")

def update(i):
    global q, arc1, arc2, centre

    delta = np.array([
                        int(s[0] and not s[1]),    # delta1
                        int(not s[0] and s[1]),    # delta2
                        int(all(s))                # delta3
                    ])
    
    J_array = np.array([get_jacobian(i, q, s) for i in range(1, 4)])
    
    # Perform matrix multiplication
    # J_array shape: (3, m, n), delta shape: (3,)
    # Result shape: (m, n)
    J =  np.tensordot(J_array, delta, axes=([0], [0]))
    
    q_dot = J@v
    q += q_dot * dt

    q[3] = round(q[3], 4)
    q[4] = round(q[4], 4)

    centre.set_data([q[0]], [q[1]])

    seg1 = arc(q, 1)
    seg2 = arc(q, 2)

    arc1.set_data(seg1[0], seg1[1])
    arc2.set_data(seg2[0], seg2[1])

    if s[0] == 0:
        arc1.set_color("blue")
    else:
        arc1.set_color("red")

    if s[1] == 0:
        arc2.set_color("blue")
    else:
        arc2.set_color("red")

    return arc1, arc2, centre

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

    # lu2_path_x = []
    # lu2_path_y = []

    # for k in range(0, 81, 1):
    #     vss1_ = arc_new(0, k, 2)

    #     origin_ = [vss1_[0][-1], vss1_[1][-1], vss1_[2]]

    #     vss2_ = arc_new(origin_[2], 0, 2)
    #     vss2_[0] += origin_[0]
    #     vss2_[1] += origin_[1]

    #     lu2_path_x.append(vss2_[0][-1])
    #     lu2_path_y.append(vss2_[1][-1])

    # plt.scatter(lu2_path_x, lu2_path_y)

    plt.show()