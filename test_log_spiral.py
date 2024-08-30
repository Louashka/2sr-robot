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

center_x = 0.0152
center_y = 0.014

spiral1 = splines.LogSpiral(1)
spiral2 = splines.LogSpiral(2)

dt = 0.1
k = 0
t = 20
counter = 0
v = 0.2

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
k_array_positive = np.linspace(0, k_max, 15)
k_array_negative = np.linspace(0, -k_max, 15)

for k in k_array_positive:
    vss1 = arc(0, k, 2)
    
    plt.plot(vss1[0], vss1[1], '-r')

    origin = [vss1[0][-1], vss1[1][-1], vss1[2]]

    vss2 = arc(origin[2], k, 2)
    vss2[0] += origin[0]
    vss2[1] += origin[1]

    plt.plot(vss2[0], vss2[1], '-b')

    

for k in k_array_negative:
    vss1 = arc(0, k, 2)
    
    plt.plot(vss1[0], vss1[1], '-r')

    origin = [vss1[0][-1], vss1[1][-1], vss1[2]]

    vss2 = arc(origin[2], k, 2)
    vss2[0] += origin[0]
    vss2[1] += origin[1]

    plt.plot(vss2[0], vss2[1], '-b')


theta = np.linspace(0, 2 * np.pi, 50)

# a = gv.L_VSS / 1.5
# b = gv.L_VSS / 1.37

# x = a * np.cos(theta)
# y = b * np.sin(theta)

# x += gv.L_VSS - a

a = 0.043
r = 2 * a * (1 - np.cos(theta))

x = -r * np.cos(theta) - 0.017
y = r * np.sin(theta)

plt.plot(x, y)

# x_top = a * np.cos(np.pi/2)
# y_top = b * np.sin(np.pi/2)

# plt.plot(x_top + gv.L_VSS - a, y_top, 'ro')

# theta = np.linspace(-1.8, 0.018, 20)

# a = 0.0634
# b = 0.2697

# r = a * np.exp(b * theta)

# x = r * np.cos(theta)
# y = r * np.sin(theta)

# x += 0.015

# print(x[-1])
# print(y[-1])
# print()

# plt.plot(x, y)

plt.axis('equal')
plt.show()

