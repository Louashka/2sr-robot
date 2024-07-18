import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from Model import global_var as gv

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

def get_spiral_parameters(n: int, k: float) -> tuple[float, float, float, float]:
    n -= 1

    a = gv.SPIRAL_COEF[0][n]
    b = gv.SPIRAL_COEF[1][n]
    theta = k * gv.L_VSS / gv.M[n]
    theta_min = gv.SPIRAL_TH_MIN[n]

    return a, b, theta, theta_min


def get_rho(n: int, k: float) -> float:
    a, b, theta, theta_min = get_spiral_parameters(n, k)

    if k > 0:
        b = -b
        theta -= theta_min
    else:
        theta += theta_min

    rho = a * np.exp(b * theta)

    return rho

def get_pos_dot(n: int, theta_0: float, k: float, scale: float, seg:int = 1, lu: int = 1) -> list:
    seg_flag = -1 if seg == 1 else 1
    lu_flag = -1 if lu == 1 else 1

    _, b, theta, _ = get_spiral_parameters(n, k)

    phi = theta_0 + seg_flag * k * gv.L_VSS
  
    pos_dot_local = np.array([[b * np.cos(theta) - np.sin(theta)],
                              [lu_flag * (b * np.sin(theta) + np.cos(theta))]])

    pos_dot_local *= scale
    
    rot_spiral_to_global = np.array([[np.cos(phi), -np.sin(phi)],
                                       [np.sin(phi), np.cos(phi)]])
    
    pos_dot_global = rot_spiral_to_global@pos_dot_local
    
    return pos_dot_global.flatten().tolist()

def get_jacobian():
    J = np.array([[0, x12_dot],
                  [0, y12_dot],
                  [0, theta12_dot],
                  [k11_dot, k12_dot],
                   [0, 0]])
    
    return


fig, ax = plt.subplots()

dt = 0.1  # step size

q = np.array([0.0, 0.0, np.pi/5, 0.0, 0.0])
s = [1, 1]

v = np.array([0.04, 0.04])

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

    #//////////////////////////////////Case I.I//////////////////////////////////

    rho11 = get_rho(1, q[3])

    k11_dot = -gv.M[0] / (gv.L_VSS * rho11)

    #//////////////////////////////////Case I.II/////////////////////////////////

    rho12 = get_rho(2, q[3])
 
    theta12_dot = gv.M[1] / rho12
    k12_dot = theta12_dot / gv.L_VSS

    scale1 = (rho12 - gv.L_VSS) / rho12
    x12_dot, y12_dot = get_pos_dot(2, q[2], q[3], scale1, 1, 2)

    #Combine case I.I and I.II into complete Case I

    J1 = np.array([[0, x12_dot],
                   [0, y12_dot],
                   [0, theta12_dot],
                   [k11_dot, k12_dot],
                   [0, 0]])
    
    #////////////////////////////////////////////////////////////////////////////

    #//////////////////////////////////Case II.I/////////////////////////////////

    rho21 = get_rho(1, q[4])

    k21_dot = gv.M[0] / (gv.L_VSS * rho21)

    #//////////////////////////////////Case II.II////////////////////////////////

    rho22 = get_rho(2, q[4])

    theta22_dot = gv.M[1] / rho22
    k22_dot = -theta22_dot / gv.L_VSS
    
    scale2 = (rho22 - gv.L_VSS) / rho22
    x22_dot, y22_dot = get_pos_dot(2, q[2], q[4], scale2, 2, 1)

    #Combine case II.I and II.II into complete Case II

    J2 = np.array([[x22_dot, 0],
                   [y22_dot, 0],
                   [theta22_dot, 0],
                   [0, 0],
                   [k22_dot, k21_dot]])
    
    #////////////////////////////////////////////////////////////////////////////

    #//////////////////////////////////Case III//////////////////////////////////

    rho3 = get_rho(3, (q[3] + q[4]) / 2)

    theta3_dot = gv.M[2] / rho3
    k3_dot = theta3_dot / gv.L_VSS
    
    scale3 = (rho3 - rho11) / rho3
    x31_dot, y31_dot = get_pos_dot(3, q[2], q[4], scale3, 2, 1)
    x32_dot, y32_dot = get_pos_dot(3, q[2], q[3], scale3, 1, 2)

    J3 = np.array([[x31_dot, x32_dot],
                   [y31_dot, y32_dot],
                   [theta3_dot, theta3_dot],
                   [-k3_dot, k3_dot],
                   [-k3_dot, k3_dot]])

    #////////////////////////////////////////////////////////////////////////////

    # Combine all parts into one Jacobian

    delta1 = int(s[0] and not s[1])
    delta2 = int(not s[0] and s[1])
    delat3 = int(all(s))
    
    J = delta1 * J1 + delta2 * J2 + delat3 * J3
    
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