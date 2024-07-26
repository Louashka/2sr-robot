import numpy as np
import matplotlib.pyplot as plt
# from scipy.optimize import fsolve
# from Controller import path
import sys
sys.path.append('/Users/lytaura/Documents/PolyU/Research/2SR/Version 1/Multi agent/Control/2sr-swarm-control')
from Model import global_var as gv
from scipy.optimize import curve_fit

# def func(var, *args):
#     k = var[0]

#     # x, y, x0, y0, theta0, l = args
#     # theta1 = theta0 + k * l
    
#     # eq1 = x - x0 - (np.sin(theta1) - np.sin(theta0)) / k
#     # eq2 = y - y0 - (np.cos(theta0) - np.cos(theta1)) / k

#     c, l = args

#     theta = k * l

#     eq = c * theta - 2 * l * np.sin(theta / 2)

#     # print(k)

#     # print(eq1)
#     # print(eq2)
#     # print("")

#     return eq

# if __name__ == "__main__":

#     k = 25
#     l = 0.12
#     theta0 = np.pi / 6

#     theta1 = theta0 - k * l

#     x0 = 0.12
#     y0 = 0.17

#     x = x0 + (np.sin(theta1) - np.sin(theta0)) / k
#     y = y0 + (np.cos(theta0) - np.cos(theta1)) / k

#     c = path.distance([x0, y0], [x, y])
#     print(c)

#     # k_solution = fsolve(func, 1, (x, y, x0, y0, theta0, l))
#     k_solution = fsolve(func, 50, (c, l))
#     print(k_solution)

#     theta_array = theta0 - k * np.linspace(0, l, 50)
#     x_array = x0 + (np.sin(theta_array) - np.sin(theta0)) / k
#     y_array = y0 + (np.cos(theta0) - np.cos(theta_array)) / k

#     plt.plot(x_array, y_array)
#     plt.plot(x, y, 'r*')
#     plt.axis('equal')
#     plt.show()

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

def logarithmic_spiral(x, ro, x0, y0):

    y = y0 + np.sqrt(ro**2 - (x - x0)**2)

    return y

def log_spiral_ctr(x, y):
    dx = np.diff(x)
    dy = np.diff(y)

    s = np.sqrt(np.square(dx) + np.square(dy))
    heading = np.unwrap(np.arctan2(dy, dx))
    dphi = np.diff(heading)

    r = np.multiply(s[:-1], np.tan(np.pi / 2 - dphi)) 

    xc = x[:-2] + np.multiply(r, np.cos(heading[:-1] + np.pi / 2))
    yc = y[:-2] + np.multiply(r, np.sin(heading[:-1] + np.pi / 2))

    xc = np.mean(xc)
    yc = np.mean(yc)

    return [xc, yc]

def func(theta, a, b):
    return a * np.exp(b * theta)

def log_spiral_coeffs(x, y, c):
    dx = x - c[0]
    dy = y - c[1]

    r = np.sqrt(np.square(dx) + np.square(dy))

    theta = np.arctan2(y, x)
    theta = np.unwrap(theta)

    th_min = np.min(theta)
    th_max = np.max(theta)

    # p = np.polyfit(theta, np.log(r), 1)

    # a_fit = np.exp(p[1])
    # b_fit = p[0]
    
    popt, pcov = curve_fit(func, theta, r)
    a_fit, b_fit = popt

    return a_fit, b_fit, th_min, th_max

colors = {'red': [0.9333, 0.25882, 0.21176], 'blue': [0.1843, 0.52157, 0.6666], 'brown': [0.337255, 0.2549, 0.2196]}
robot_lw = 5
spiral_lw = 2
spiral_edge_lw = 2
circle_marker = 7
hollow_marker = 10
center_marker = 9

step_size = 1
theta_n = 40

# origin = [0, 0]

# vss1 = arc(0, 0)

# lu2_path_x = []
# lu2_path_y = []

# for k in range(0, 81, step_size):
#     vss2 = arc(0, k, 2)

#     lu2_path_x.append(vss2[0][-1])
#     lu2_path_y.append(vss2[1][-1])

# x_array = np.array(lu2_path_x)
# y_array = np.array(lu2_path_y)

# spiral_center = log_spiral_ctr(x_array, y_array)
# a_fit, b_fit, th_min, th_max = log_spiral_coeffs(x_array, y_array, spiral_center)

# th_min -= 0.18
# th_max += 0.6

# print(f'I  spiral: a = {a_fit:.4f}')
# print(f'I  spiral: b = {b_fit:.4f}')
# print(f'I  spiral: theta_min = {th_min:.4f}')
# print(f'I  spiral: theta_max = {th_max:.4f}')
# print(f'I  spiral: center x = {spiral_center[0]:.4f}')
# print(f'I  spiral: center y = {spiral_center[1]:.4f}')
# print("")

# # popt, pcov = curve_fit(logarithmic_spiral, lu2_path_x, lu2_path_y)
# # ro_fit, x0_fit, y0_fit = popt

# theta_array = np.linspace(th_min, th_max, theta_n)

# ro_fit = a_fit * np.exp(b_fit * theta_array)
# log_spiral_x = spiral_center[0] + ro_fit * np.cos(theta_array)
# log_spiral_y = spiral_center[1] + ro_fit * np.sin(theta_array)

# f, axs = plt.subplots(3, 1, figsize=(5, 8))

# for x, y in zip(log_spiral_x, log_spiral_y):
#     axs[0].plot([spiral_center[0], x], [spiral_center[1], y], color=colors['red'], lw=spiral_lw, alpha=0.5)

# axs[0].plot(spiral_center[0], spiral_center[1], 'o', color=colors['brown'], markersize=center_marker)

# axs[0].plot(vss1[0], vss1[1], color=colors['blue'], lw=robot_lw)
# axs[0].plot(vss1[0][-1], vss1[1][-1], 'o', color=colors['blue'], markersize=circle_marker)

# axs[0].plot(vss2[0], vss2[1], color=colors['red'], lw=robot_lw)
# axs[0].plot(vss2[0][-1], vss2[1][-1], 'o', color=colors['red'], markersize=circle_marker)

# axs[0].scatter(lu2_path_x, lu2_path_y, facecolors='none', edgecolors=colors['red'], s=hollow_marker, lw=spiral_edge_lw)

# axs[0].plot(0, 0, '*', color=colors['brown'], markersize=center_marker)

# axs[0].axis('equal')
# axs[0].tick_params(axis='x', labelsize=20)
# axs[0].tick_params(axis='y', labelsize=20)
# # # /////////////////////////////////////////////////////

# vss1 = arc(0, 41, 2)

# origin = [vss1[0][-1], vss1[1][-1], vss1[2]]

# vss2 = arc(origin[2], 0, 2)
# vss2[0] += origin[0]
# vss2[1] += origin[1]

# lu2_path_x = []
# lu2_path_y = []

# for k in range(0, 81, step_size):
#     vss1_ = arc(0, k, 2)

#     origin_ = [vss1_[0][-1], vss1_[1][-1], vss1_[2]]

#     vss2_ = arc(origin_[2], 0, 2)
#     vss2_[0] += origin_[0]
#     vss2_[1] += origin_[1]

#     lu2_path_x.append(vss2_[0][-1])
#     lu2_path_y.append(vss2_[1][-1])

# x_array = np.array(lu2_path_x)
# y_array = np.array(lu2_path_y)

# spiral_center = log_spiral_ctr(x_array, y_array)
# a_fit, b_fit, th_min, th_max = log_spiral_coeffs(x_array, y_array, spiral_center)

# th_min -= 0.1
# th_max -= 0.25

# print(f'II  spiral: a = {a_fit:.4f}')
# print(f'II  spiral: b = {b_fit:.4f}')
# print(f'II  spiral: theta_min = {th_min:.4f}')
# print(f'II  spiral: theta_max = {th_max:.4f}')
# print(f'II  spiral: center x = {spiral_center[0]:.4f}')
# print(f'II  spiral: center y = {spiral_center[1]:.4f}')
# print("")

# theta_array = np.linspace(th_min, th_max, theta_n)

# ro_fit = a_fit * np.exp(b_fit * theta_array)
# log_spiral_x = spiral_center[0] + ro_fit * np.cos(theta_array)
# log_spiral_y = spiral_center[1] + ro_fit * np.sin(theta_array)

# for x, y in zip(log_spiral_x, log_spiral_y):
#     axs[1].plot([spiral_center[0], x], [spiral_center[1], y], color=colors['blue'], lw=spiral_lw, alpha=0.5)

# axs[1].plot(spiral_center[0], spiral_center[1], 'o', color=colors['brown'], markersize=center_marker)

# axs[1].plot(vss1[0], vss1[1], color=colors['red'], lw=robot_lw)
# axs[1].plot(vss1[0][0], vss1[1][0], 'o', color=colors['red'], markersize=circle_marker)

# axs[1].plot(vss2[0], vss2[1], color=colors['blue'], lw=robot_lw)
# axs[1].plot(vss2[0][-1], vss2[1][-1], 'o', color=colors['blue'], markersize=circle_marker)

# axs[1].scatter(lu2_path_x, lu2_path_y, facecolors='none', edgecolors=colors['blue'], s=hollow_marker, lw=spiral_edge_lw)

# axs[1].plot(origin[0], origin[1], '*k', markersize=center_marker)
# axs[1].axis('equal')
# axs[1].tick_params(axis='x', labelsize=20)
# axs[1].tick_params(axis='y', labelsize=20)
# # # /////////////////////////////////////////////////////

# vss1 = arc(0, 41, 2)

# origin = [vss1[0][-1], vss1[1][-1], vss1[2]]

# vss2 = arc(origin[2], 41, 2)
# vss2[0] += origin[0]
# vss2[1] += origin[1]

# lu2_path_x = []
# lu2_path_y = []

# for k in range(0, 41, 1):
#     vss1_ = arc(0, k, 2)

#     origin_ = [vss1_[0][-1], vss1_[1][-1], vss1_[2]]

#     vss2_ = arc(origin_[2], k, 2)
#     vss2_[0] += origin_[0]
#     vss2_[1] += origin_[1]

#     lu2_path_x.append(vss2_[0][-1])
#     lu2_path_y.append(vss2_[1][-1])

# x_array = np.array(lu2_path_x)
# y_array = np.array(lu2_path_y)

# spiral_center = log_spiral_ctr(x_array, y_array)
# a_fit, b_fit, th_min, th_max = log_spiral_coeffs(x_array, y_array, spiral_center)

# th_min -= 0.19
# th_max += 0.61

# print(f'III  spiral: a = {a_fit:.4f}')
# print(f'III  spiral: b = {b_fit:.4f}')
# print(f'III  spiral: theta_min = {th_min:.4f}')
# print(f'III  spiral: theta_max = {th_max:.4f}')
# print(f'III  spiral: center x = {spiral_center[0]:.4f}')
# print(f'III  spiral: center y = {spiral_center[1]:.4f}')

# theta_array = np.linspace(th_min, th_max, theta_n)

# ro_fit = a_fit * np.exp(b_fit * theta_array)
# log_spiral_x = spiral_center[0] + ro_fit * np.cos(theta_array)
# log_spiral_y = spiral_center[1] + ro_fit * np.sin(theta_array)

# for x, y in zip(log_spiral_x, log_spiral_y):
#     axs[2].plot([spiral_center[0], x], [spiral_center[1], y], color=colors['red'], lw=spiral_lw, alpha=0.5)

# axs[2].plot(spiral_center[0], spiral_center[1], 'o', color=colors['brown'], markersize=center_marker)

# axs[2].plot(vss1[0], vss1[1], color=colors['red'], lw=robot_lw)
# axs[2].plot(vss1[0][0], vss1[1][0], 'o', color=colors['red'], markersize=circle_marker)

# axs[2].plot(vss2[0], vss2[1], color=colors['red'], lw=robot_lw)
# axs[2].plot(vss2[0][-1], vss2[1][-1], 'o', color=colors['red'], markersize=circle_marker)

# axs[2].scatter(lu2_path_x, lu2_path_y, facecolors='none', edgecolors=colors['red'], s=hollow_marker, lw=spiral_edge_lw)

# axs[2].plot(origin[0], origin[1], '*k', markersize=center_marker)

# axs[2].axis('equal')
# axs[2].tick_params(axis='x', labelsize=20)
# axs[2].tick_params(axis='y', labelsize=20)

a =  0.1166
b = -0.1764

# th_min = -0.058 * np.pi
# th_max = 0.58 * np.pi

th_min = -0.19
th_max = 1.95

center_x = 0.0347
center_y = 0.0225

theta_array = np.linspace(th_min, th_max, theta_n, endpoint=True)

ro = a * np.exp(b * theta_array)
log_spiral_x = center_x + ro * np.cos(theta_array)
log_spiral_y = center_y + ro * np.sin(theta_array)

plt.scatter(-log_spiral_x, log_spiral_y)


for k in range(0, 21, 1):
    # vss2 = arc(0, k, 2)

    vss1 = arc(0, k, 1)
    plt.plot(vss1[0], vss1[1])

    origin = [vss1[0][-1], vss1[1][-1], vss1[2]]

    vss2 = arc(origin[2], k, 1)
    vss2[0] += origin[0]
    vss2[1] += origin[1]
    
    plt.plot(vss2[0], vss2[1])

plt.axis('equal')

plt.show()

# f.savefig('log_spiral_new.png', dpi=600, transparent=True)


