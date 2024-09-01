import numpy as np
import matplotlib.pyplot as plt
from Model import global_var as gv, splines

def arc(theta0, k, seg=1):
    l = np.linspace(0, gv.L_VSS, 50)
    flag = -1 if seg == 1 else 1
    theta_array = theta0 + flag * k * l

    if k == 0:
        x = np.array([0, flag * gv.L_VSS * np.cos(theta0)])
        y = np.array([0, flag * gv.L_VSS * np.sin(theta0)])
    else:
        x = np.sin(theta_array) / k - np.sin(theta0) / k
        y = -np.cos(theta_array) / k + np.cos(theta0) / k

    theta = theta_array[-1]
        
    return [x, y, theta % (2 * np.pi)]


dt = 0.1
k = 0
t = 20
counter = 0
v = 0.97

# while counter < t:
#     vss1 = arc(0, k, 2)

#     origin = [vss1[0][-1], vss1[1][-1], vss1[2]]

#     vss2 = arc(origin[2], 0, 2)
#     vss2[0] += origin[0]
#     vss2[1] += origin[1]

#     x, y = spiral2.get_pos(k)
#     x += center_x
#     y += center_y

#     plt.plot(vss1[0], vss1[1], '-r')
#     plt.plot(vss2[0], vss2[1], '-b')

#     plt.plot(x, y, '.m')

#     x, y = spiral1.get_pos(k)
#     x += 0.0115
#     y += 0.014

#     plt.plot(x, y, '.m')

#     k += spiral2.get_k_dot(k) * v * dt
#     counter += 1


# plt.plot(center_x, center_y, '*b')
   
# plt.axis('equal')
# plt.show()

k_max = np.pi / gv.L_VSS
k_array = np.linspace(-k_max, k_max, 30)

cardiod = splines.Cardioid(1)

vss1 = arc(0, k_array[0], 2)
x_, y_, th = vss1[0][-1], vss1[1][-1], vss1[2]

for k in k_array:
    vss1 = arc(0, k, 2)
    th = vss1[2]
    print(th)
    
    plt.plot(vss1[0], vss1[1], '-r')

    # origin = [vss1[0][-1], vss1[1][-1], vss1[2]]

    # vss2 = arc(origin[2], k, 2)
    # vss2[0] += origin[0]
    # vss2[1] += origin[1]

    # plt.plot(vss2[0], vss2[1], '-b')

    x, y = cardiod.pos(k, 2)
    x -= 0.0085

    plt.plot(x, y, '.m')

    # Generate points based on pos_dot
    # dx, dy = cardiod.pos_dot(th, k, 1, 2)

    # x_ += dx * v * dt
    # y_ += dy * v * dt

    # plt.plot(x_, y_, '.g')


plt.axis('equal')
plt.show()

