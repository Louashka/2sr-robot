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
        
    return [100 * x, 100 * y, theta % (2 * np.pi)]


def cardioid(i, lu=1):
    rho = 100 * 2 * gv.CARDIOID_A[i] * (1 - np.cos(phi1))
    x = rho * np.cos(phi1) + 100 * gv.CARDIOID_OFFSET[i]
    y = rho * np.sin(phi1)

    if lu == 2:
        x = -x

    return x, y

# Create a figure with 3 subplots in a row
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
font_s = 14

lw = 2
marker_s_in = 10
marker_s_out = 15

phi1 = np.linspace(0, 2*np.pi, 40) 
phi2 = np.linspace(0, 2*np.pi, 40) 
phi3 = np.linspace(0, 2*np.pi, 40) 

k_max = np.pi / (2*gv.L_VSS)
k_array = np.linspace(-k_max, k_max, 10)


tan = '#CAB1A3'
red = '#c44536'
blue = '#308185'
grey = '#283d3b'

plt.rcParams.update({'font.size': font_s})

for ax in axs:
    # ax.axis('off')
    ax.set_xlabel('x [cm]', fontsize=font_s)
    if ax == axs[0]:
        ax.set_ylabel('y [cm]', fontsize=font_s)
    ax.tick_params(axis='both', which='major', labelsize=font_s)
    ax.axis('equal')

# -------------------------------------------------------------------
alpha = 0.45
label = None

vss2 = arc(0, 0, 2)
axs[0].plot(vss2[0], vss2[1], color=tan, lw = lw, label='rigid VSS')

for k in k_array:
    if k == k_array[-1]:
        alpha = 1
        label = 'soft VSS'

    vss1 = arc(0, k, 1)
    axs[0].plot(vss1[0], vss1[1], color=red, lw = lw, alpha=alpha, label=label)


x, y = cardioid(0)
axs[0].scatter(x, y, color=grey, s=marker_s_out, zorder=2)
axs[0].legend()

# -------------------------------------------------------------------
alpha = 0.45

for k in k_array:
    if k == k_array[-1]:
        alpha = 1

    vss1 = arc(0, k, 2) 
    axs[1].plot(vss1[0], vss1[1], color=red, lw = lw, alpha=alpha)

    origin = [vss1[0][-1], vss1[1][-1], vss1[2]]

    vss2 = arc(origin[2], 0, 2)
    vss2[0] += origin[0]
    vss2[1] += origin[1]
    axs[1].plot(vss2[0], vss2[1], color=tan, lw = lw, alpha=alpha)

x, y = cardioid(0, 2)
axs[1].scatter(x, y, color=blue, s=marker_s_in, zorder=2)

x, y = cardioid(1, 2)
axs[1].scatter(x, y, color=grey, s=marker_s_out, zorder=2)

# -------------------------------------------------------------------
alpha = 0.45

for k in k_array:
    if k == k_array[-1]:
        alpha = 1

    vss2 = arc(0, k, 1)
    axs[2].plot(vss2[0], vss2[1], color=red, lw = lw, alpha=alpha)

    origin = [vss2[0][-1], vss2[1][-1], vss2[2]]

    vss1 = arc(origin[2], k, 1)
    vss1[0] += origin[0]
    vss1[1] += origin[1]
    axs[2].plot(vss1[0], vss1[1], color=red, lw = lw, alpha=alpha)

    axs[2].scatter([origin[0]], [origin[1]], color=blue, s=marker_s_in, zorder=2)

x, y = cardioid(2)
axs[2].scatter(x, y, color=grey, s=marker_s_out, zorder=2)

plt.savefig('Experiments/Figures/cardioid_figures.pdf', dpi=300, bbox_inches='tight')  # Save the figure as a PNG file with high resolution



plt.tight_layout()  # Adjust layout
plt.show()





