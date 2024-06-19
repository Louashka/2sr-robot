import sys
sys.path.append('D:/Robot 2SR/2sr-swarm-control')
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Rectangle
from Model import global_var 

link_width = global_var.D_BRIDGE
link_length = global_var.L_CONN

LINK_DIAG = ((link_length / 2)**2 + (link_width / 2)**2)**(1 / 2)

font_size = 22
fig, ax = plt.subplots()
plt.xticks(fontsize = font_size)
plt.yticks(fontsize = font_size)

q_list = []
q_target = []
s_array = []

head = []
tail = []

wheel1 = []
wheel2 = []
wheel3 = []
wheel4 = []

x_range = 0
y_range = 0

# link = Rectangle((0, 0), 0, 0, fc='y')
arc1, = ax.plot([], [], lw=5, color="blue")
arc2, = ax.plot([], [], lw=5, color="blue")
centre, = ax.plot([], [], lw=5, marker=".", color="black")

stiffness_text = ax.text(0, 0, '', fontsize=font_size)

target_arc1, = ax.plot([], [], lw=3, color="black", alpha=0.3)
target_arc2, = ax.plot([], [], lw=3, color="black", alpha=0.3)

head_block = Rectangle((0, 0), 0, 0, fc='k')
tail_block = Rectangle((0, 0), 0, 0, fc='k')

wheel1_block = Rectangle((0, 0), 0, 0, fc='r')
wheel2_block = Rectangle((0, 0), 0, 0, fc='y')
wheel3_block = Rectangle((0, 0), 0, 0, fc='m')
wheel4_block = Rectangle((0, 0), 0, 0, fc='g')

wheel1_direction, = ax.plot([], [], lw=2, color="red")
wheel2_direction, = ax.plot([], [], lw=2, color="red")
wheel3_direction, = ax.plot([], [], lw=2, color="red")
wheel4_direction, = ax.plot([], [], lw=2, color="red")

agent_direction, = ax.plot([], [], lw=3, color="red")
head_direction, = ax.plot([], [], lw=3, color="red")
tail_direction, = ax.plot([], [], lw=3, color="red")

# ax.axis('off')


def init():
    global ax, x_range, y_range    

    x_range, y_range = defineRange()
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_aspect("equal")

    ax.add_patch(head_block)
    ax.add_patch(tail_block)

    ax.add_patch(wheel1_block)
    ax.add_patch(wheel2_block)
    ax.add_patch(wheel3_block)
    ax.add_patch(wheel4_block)

    # ax.add_patch(link)

    # if q_target:
    #     ax.add_patch(target_link)


def defineRange():
    margin = 0.5

    q_array = np.array(q_list)
    x_min, y_min = q_array[:, :2].min(axis=0)
    x_max, y_max = q_array[:, :2].max(axis=0)

    ax_range = max(x_max - x_min, y_max - y_min) + margin

    x_range = (x_min - margin, x_min + ax_range)
    y_range = (y_min - margin, y_min + ax_range)

    return x_range, y_range


def genArc(q, seg):
    s = np.linspace(0, global_var.L_VSS, 50)

    flag = -1 if seg == 1 else 1

    gamma_array = q[2] + flag * q[2 + seg] * s

    x_0 = q[0]
    y_0 = q[1]

    if q[2 + seg] == 0:
        x = x_0 + [0, flag * global_var.L_VSS * np.cos(q[2])]
        y = y_0 + [0, flag * global_var.L_VSS * np.sin(q[2])]
    else:
        x = x_0 + np.sin(gamma_array) / \
            q[2 + seg] - np.sin(q[2]) / q[2 + seg]
        y = y_0 - np.cos(gamma_array) / \
            q[2 + seg] + np.cos(q[2]) / q[2 + seg]

    return [x, y]


def update(i):
    global arc1, arc2, centre, stiffness_text, target_arc1, target_arc2, head_block, tail_block
    global wheel1_block, wheel2_block, wheel3_block, wheel4_block, agent_direction, head_direction, tail_direction
    global wheel1_direction, wheel2_direction, wheel3_direction, wheel4_direction
    q = q_list[i]

    x = q[0]
    y = q[1]
    phi = q[2]

    # x = 0
    # y = 0
    # phi = 0

    # x0 = x - link_length / 2
    # y0 = y - link_width / 2

    head_x = head[i][0]
    head_y = head[i][1]
    head_theta = head[i][2]

    head_block.set_width(global_var.LU_SIDE)
    head_block.set_height(global_var.LU_SIDE)
    head_block.set_xy([head_x - global_var.LU_SIDE/2, head_y - global_var.LU_SIDE/2])

    transform = mpl.transforms.Affine2D().rotate_around(
        head_x, head_y, head_theta) + ax.transData
    head_block.set_transform(transform)

    tail_x = tail[i][0]
    tail_y = tail[i][1]
    tail_theta = tail[i][2]

    tail_block.set_width(global_var.LU_SIDE)
    tail_block.set_height(global_var.LU_SIDE)
    tail_block.set_xy([tail_x - global_var.LU_SIDE/2, tail_y - global_var.LU_SIDE/2])

    transform = mpl.transforms.Affine2D().rotate_around(
        tail_x, tail_y, tail_theta) + ax.transData
    tail_block.set_transform(transform)

    #//////////////////////////////////////////////////////////
    wheel1_x = wheel1[i][0]
    wheel1_y = wheel1[i][1]
    # wheel1_theta = wheel1[i][2]
    wheel1_theta = head_theta + global_var.BETA[0]

    wheel1_block.set_width(2 * global_var.WHEEL_R)
    wheel1_block.set_height(global_var.WHEEL_TH)
    wheel1_block.set_xy([wheel1_x - global_var.WHEEL_R, wheel1_y - global_var.WHEEL_TH/2])

    transform = mpl.transforms.Affine2D().rotate_around(
        wheel1_x, wheel1_y, wheel1_theta) + ax.transData
    wheel1_block.set_transform(transform)

    wheel1_direction.set_data([wheel1_x, wheel1_x + 0.05 * np.cos(wheel1_theta)], 
                              [wheel1_y, wheel1_y + 0.05 * np.sin(wheel1_theta)])


    wheel2_x = wheel2[i][0]
    wheel2_y = wheel2[i][1]
    # wheel2_theta = wheel2[i][2]
    wheel2_theta = head_theta + global_var.BETA[1]

    wheel2_block.set_width(2 * global_var.WHEEL_R)
    wheel2_block.set_height(global_var.WHEEL_TH)
    wheel2_block.set_xy([wheel2_x - global_var.WHEEL_R, wheel2_y - global_var.WHEEL_TH/2])

    transform = mpl.transforms.Affine2D().rotate_around(
        wheel2_x, wheel2_y, wheel2_theta) + ax.transData
    wheel2_block.set_transform(transform)

    wheel2_direction.set_data([wheel2_x, wheel2_x + 0.05 * np.cos(wheel2_theta)], 
                              [wheel2_y, wheel2_y + 0.05 * np.sin(wheel2_theta)])

    
    wheel3_x = wheel3[i][0]
    wheel3_y = wheel3[i][1]
    # wheel3_theta = wheel3[i][2]
    wheel3_theta = tail_theta + global_var.BETA[2]

    wheel3_block.set_width(2 * global_var.WHEEL_R)
    wheel3_block.set_height(global_var.WHEEL_TH)
    wheel3_block.set_xy([wheel3_x - global_var.WHEEL_R, wheel3_y - global_var.WHEEL_TH/2])

    transform = mpl.transforms.Affine2D().rotate_around(
        wheel3_x, wheel3_y, wheel3_theta) + ax.transData
    wheel3_block.set_transform(transform)

    wheel3_direction.set_data([wheel3_x, wheel3_x + 0.05 * np.cos(wheel3_theta)], 
                              [wheel3_y, wheel3_y + 0.05 * np.sin(wheel3_theta)])


    wheel4_x = wheel4[i][0]
    wheel4_y = wheel4[i][1]   
    # wheel4_theta = wheel4[i][2]
    wheel4_theta = tail_theta + global_var.BETA[3]

    wheel4_block.set_width(2 * global_var.WHEEL_R)
    wheel4_block.set_height(global_var.WHEEL_TH)
    wheel4_block.set_xy([wheel4_x - global_var.WHEEL_R, wheel4_y - global_var.WHEEL_TH/2])

    transform = mpl.transforms.Affine2D().rotate_around(
        wheel4_x, wheel4_y, wheel4_theta) + ax.transData
    wheel4_block.set_transform(transform)

    wheel4_direction.set_data([wheel4_x, wheel4_x + 0.05 * np.cos(wheel4_theta)], 
                              [wheel4_y, wheel4_y + 0.05 * np.sin(wheel4_theta)])

    
    agent_direction.set_data([x, x + 0.1 * np.cos(phi)], 
                             [y, y + 0.1 * np.sin(phi)])
    
    head_direction.set_data([head_x, head_x + 0.1 * np.cos(head_theta)], 
                            [head_y, head_y + 0.1 * np.sin(head_theta)])
    
    tail_direction.set_data([tail_x, tail_x + 0.1 * np.cos(tail_theta)], 
                            [tail_y, tail_y + 0.1 * np.sin(tail_theta)])

    #//////////////////////////////////////////////////////////

    # seg1 = genArc(q, 1)
    # seg2 = genArc(q, 2)

    seg1 = genArc([x, y, phi, q[3], q[4]], 1)
    seg2 = genArc([x, y, phi, q[3], q[4]], 2)

    arc1.set_data(seg1[0], seg1[1])
    arc2.set_data(seg2[0], seg2[1])

    if s_array[i][0] == 0:
        arc1.set_color("blue")
    else:
        arc1.set_color("red")

    if s_array[i][1] == 0:
        arc2.set_color("blue")
    else:
        arc2.set_color("red")

    centre.set_data([x], [y])

    # stiffness_text.set_text(
    #     "s1: " + str(s_array[i][0]) + ", s2: " + str(s_array[i][1]))
    # stiffness_text.set_position(
    #     (x_range[1] - (x_range[1] - x_range[0]) / 3, y_range[1] - (y_range[1] - y_range[0]) / 15))
    # stiffness_text.set_position(
    #     (x_range[0] + (x_range[1] - x_range[0]) / 25, y_range[0] + (y_range[1] - y_range[0]) / 40))

    if q_target:

        target_seg1 = genArc(q_target, 1)
        target_seg2 = genArc(q_target, 2)

        target_arc1.set_data(target_seg1[0], target_seg1[1])
        target_arc2.set_data(target_seg2[0], target_seg2[1])

        return arc1, arc2, centre, stiffness_text, target_arc1, target_arc2,

    return arc1, arc2, centre, stiffness_text, head_block, tail_block


def plotMotion(q, s, h, t, wheels, frames, q_t=[]):
    global q_list, s_array, q_target, head, tail, wheel1, wheel2, wheel3, wheel4
    q_list = q
    s_array = s
    q_target = q_t
    head = h
    tail = t

    wheel1 = wheels[0]
    wheel2 = wheels[1]
    wheel3 = wheels[2]
    wheel4 = wheels[3]

    anim = FuncAnimation(fig, update, frames,
                         init_func=init, interval=1, repeat=True)

    # Save animation
    mywriter = FFMpegWriter(fps=30)
    anim.save('Experiments/Figures/reach_target1.mp4', writer=mywriter, dpi=300)
      
    plt.show()


if __name__ == "__main__":
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, 'Data/reach_target.csv')

    df = pd.read_csv(filename, header=0)
    
    column_names = ["x_target", "y_target", "angle_target", "k1_target", "k2_target", 
                       "x", "y", "angle", "k1", "k2",
                       'x_head', 'y_head', 'theta_head', 
                       'x_tail', 'y_tail', 'theta_tail',
                    #    'w1_x', 'w1_y', 'w1_theta',
                    #    'w2_x', 'w2_y', 'w2_theta',
                    #    'w3_x', 'w3_y', 'w3_theta',
                    #    'w4_x', 'w4_y', 'w4_theta',
                       'w1_x', 'w1_y',
                       'w2_x', 'w2_y',
                       'w3_x', 'w3_y',
                       'w4_x', 'w4_y',
                       ]
    df[column_names] = df[column_names].astype(float)
    
    q = [[df.iloc[i]["x"], df.iloc[i]["y"], df.iloc[i]["angle"], df.iloc[i]["k1"], df.iloc[i]["k2"]] for i in range(len(df))]
    frames_number = len(q)
    s = [[0, 0]]  * frames_number
    q_target = [df["x_target"][1], df["y_target"][1], df["angle_target"][1], df["k1_target"][1], df["k2_target"][1]]

    h = [[df.iloc[i]["x_head"], df.iloc[i]["y_head"], df.iloc[i]["theta_head"]] for i in range(len(df))]
    t = [[df.iloc[i]["x_tail"], df.iloc[i]["y_tail"], df.iloc[i]["theta_tail"]] for i in range(len(df))]

    wheels = [0] * 4

    # wheels[0] = [[df.iloc[i]["w1_x"], df.iloc[i]["w1_y"], df.iloc[i]["w1_theta"]] for i in range(len(df))]
    # wheels[1] = [[df.iloc[i]["w2_x"], df.iloc[i]["w2_y"], df.iloc[i]["w2_theta"]] for i in range(len(df))]
    # wheels[2] = [[df.iloc[i]["w3_x"], df.iloc[i]["w3_y"], df.iloc[i]["w3_theta"]] for i in range(len(df))]
    # wheels[3] = [[df.iloc[i]["w4_x"], df.iloc[i]["w4_y"], df.iloc[i]["w4_theta"]] for i in range(len(df))]
    wheels[0] = [[df.iloc[i]["w1_x"], df.iloc[i]["w1_y"]] for i in range(len(df))]
    wheels[1] = [[df.iloc[i]["w2_x"], df.iloc[i]["w2_y"]] for i in range(len(df))]
    wheels[2] = [[df.iloc[i]["w3_x"], df.iloc[i]["w3_y"]] for i in range(len(df))]
    wheels[3] = [[df.iloc[i]["w4_x"], df.iloc[i]["w4_y"]] for i in range(len(df))]

    # plotMotion(q, s, h, t, frames_number, q_t=q_target)
    plotMotion(q, s, h, t, wheels, frames_number, q_t=q_target)

