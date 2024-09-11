import sys
sys.path.append('D:/Robot 2SR/2sr-swarm-control')
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter

def parse_data(data: dict):
    timestamps = []
    temperature = []
    stiffness = []

    config_x_errors = []
    config_y_errors = []
    config_theta_errors = []
    config_k1_errors = []
    config_k2_errors = []

    vel1_errors = []
    vel2_errors = []

    distance_to_target = []

    last_temp = 22

    robot_tracking = data['robot_tracking']

    for data_entry in robot_tracking:
        stiff_trans_data = data_entry['stiff_trans']
        if stiff_trans_data['temp']:
            time_n = len(stiff_trans_data['time'])
            time_shifted = np.array(stiff_trans_data['time']) + timestamps[-1]
            timestamps += time_shifted.tolist()
            temperature += stiff_trans_data['temp']

            stiffness += [stiffness[-1]] * time_n
            config_x_errors += [config_x_errors[-1]] * time_n
            config_y_errors += [config_y_errors[-1]] * time_n
            config_theta_errors += [config_theta_errors[-1]] * time_n
            config_k1_errors += [config_k1_errors[-1]] * time_n
            config_k2_errors += [config_k2_errors[-1]] * time_n
            
            vel1_errors += [vel1_errors[-1]] * time_n
            vel2_errors += [vel2_errors[-1]] * time_n

            distance_to_target += [distance_to_target[-1]] * time_n

            last_temp = temperature[-1]

        timestamps.append(data_entry['time'])
        temperature.append(last_temp)
        stiffness.append(data_entry['config']['stiff1'])

        config_x_errors.append(data_entry['errors']['pose_errors']['e_x'])
        config_y_errors.append(data_entry['errors']['pose_errors']['e_y'])
        config_theta_errors.append(data_entry['errors']['pose_errors']['e_theta'])
        config_k1_errors.append(data_entry['errors']['pose_errors']['e_k1'])
        config_k2_errors.append(data_entry['errors']['pose_errors']['e_k2'])

        vel1_errors.append(data_entry['errors']['vel_errors']['e_v_1']) 
        vel2_errors.append(data_entry['errors']['vel_errors']['e_v_2'])

        distance_to_target.append(data_entry['distance_to_target'])

    return (timestamps, temperature, stiffness, 
            {'e_x': config_x_errors, 'e_y': config_y_errors, 
             'e_theta': config_theta_errors, 'e_k1': config_k1_errors, 
             'e_k2': config_k2_errors},
            {'e_v_1': vel1_errors, 'e_v_2': vel2_errors},
            distance_to_target)


def animate(data):
    # Apply the style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial'],
        'font.size': 10,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 10,
        'lines.linewidth': 2
    })

    # Define a color-blind friendly palette
    colors = ['#0b81a2', '#e25759', '#7f7f7f', '#e377c2', '#8c564b', '#9467bd']

    # Function to style axes
    def style_axis(ax, title, x_ticks_status=True, grid_status=True):
        ax.set_title(title, fontweight='bold', pad=15)
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='both', length=0)
        if not x_ticks_status:
            ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

        if grid_status:
            ax.grid(True, linestyle='--', alpha=0.7)
        else:
            ax.grid(False)

    # Unpack the data
    timestamps, temperature, stiffness, config_errors, vel_errors, distance_to_target = data

    # Create the figure and subplots
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    # Create twin axis for stiffness
    ax1_twin = ax1.twinx()

    # Create subplots for config errors
    ax2_sub = [ax2.inset_axes([0, i*0.2, 1, 0.17]) for i in range(5)]

    # Create subplots for velocity errors
    ax3_sub = [ax3.inset_axes([0, i*0.52, 1, 0.48]) for i in range(2)]

    # Remove ticks and labels from main ax2 and ax3
    ax2.axis('off')
    ax3.axis('off')

    ax1_twin.spines[:].set_visible(False)
    style_axis(ax1_twin, '', grid_status=False)
    ax1_twin.set_yticks([0, 1])
    # ax3_sub[1].set_xticks([])

    # for i in range(1, len(ax2_sub)):
    #     ax2_sub[i].set_xticks([])

    # Initialize empty lines
    line1, = ax1.plot([], [], color=colors[0], label='Temperature')
    line1_twin, = ax1_twin.plot([], [], lw=1, color=colors[1], label='Stiffness', linestyle='--')
    lines2 = [ax.plot([], [], color=colors[i % len(colors)])[0] for i, ax in enumerate(ax2_sub)]
    lines3 = [ax.plot([], [], color=colors[i % len(colors)])[0] for i, ax in enumerate(ax3_sub)]
    line4, = ax4.plot([], [], color=colors[0])

    # Set titles and labels
    style_axis(ax1, 'Temperature and Stiffness Change')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Temperature [°C]', color=colors[0])
    ax1_twin.set_ylabel('Stiffness', color=colors[1])

    style_axis(ax2, 'Configuration Errors', grid_status=False)
    ax2_sub[0].set_xlabel('Time [s]')

    style_axis(ax3, 'Velocity Errors', grid_status=False)
    ax3_sub[0].set_xlabel('Time [s]')

    style_axis(ax4, 'Distance to Target')
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Normalized Distance')

    # Set axis labels for config error subplots
    for ax, label in zip(ax2_sub, [r'$e_x$', r'$e_y$', r'$e_{\theta}$', r'$e_{k_1}$', r'$e_{k_2}$']):
        ax.set_ylabel(label)
        if label == r'$e_x$':
            style_axis(ax, '')
        else:
            style_axis(ax, '', x_ticks_status=False)

    # Set axis labels for velocity error subplots
    for ax, label in zip(ax3_sub, [r'$e_{v_1}$', r'$e_{v_2}$']):
        ax.set_ylabel(label)
        if label == r'$e_{v_1}$':
            style_axis(ax, '')
        else:
            style_axis(ax, '', x_ticks_status=False)

    # Add legend to ax1
    lines = [line1, line1_twin]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')

    def init():
        for ax in [ax1, ax4] + ax2_sub + ax3_sub:
            ax.set_xlim(min(timestamps), max(timestamps))
        ax1.set_ylim(min(temperature), max(temperature)+2)
        ax1_twin.set_ylim(-0.1, 1.2)  # Set y-limits for stiffness
        for ax, errors in zip(ax2_sub, config_errors.values()):
            ax.set_ylim(min(errors)-0.015, max(errors)+0.015)
        for ax, errors in zip(ax3_sub, vel_errors.values()):
            ax.set_ylim(min(errors)-0.01, max(errors)+0.01)
        ax4.set_ylim(min(distance_to_target)-0.1, max(distance_to_target)+0.1)
        return [line1, line1_twin] + lines2 + lines3 + [line4]

    def update(frame):
        # Update temperature
        line1.set_data(timestamps[:frame], temperature[:frame])

        # Update stiffness
        line1_twin.set_data(timestamps[:frame], stiffness[:frame])

        # Update config errors
        for line, errors in zip(lines2, config_errors.values()):
            line.set_data(timestamps[:frame], errors[:frame])

        # Update velocity errors
        for line, errors in zip(lines3, vel_errors.values()):
            line.set_data(timestamps[:frame], errors[:frame])

        # Update distance to target
        line4.set_data(timestamps[:frame], distance_to_target[:frame])

        return [line1, line1_twin] + lines2 + lines3 + [line4]

    # Calculate frame interval based on timestamps
    frame_interval = (timestamps[-1] - timestamps[0]) / len(timestamps) * 1000  # in milliseconds

    anim = FuncAnimation(fig, update, frames=len(timestamps), init_func=init, blit=True, interval=frame_interval)

    plt.tight_layout()
    return anim


if __name__ == "__main__":
    json_file_names = glob.glob('Experiments/Data/Tracking/SM1/*.json')

    i = 1

    for json_file_name in json_file_names:
        json_file = open(json_file_name)
        data = json.load(json_file)

        parsed_data = parse_data(data)
        anim = animate(parsed_data)

        t = parsed_data[0][-1]
        n = len(parsed_data[0])

        mywriter = FFMpegWriter(fps=int(n/t))
        anim.save(f'Experiments/Video/Animation/sm1_anim_{i}.mp4', writer=mywriter, dpi=300)

        i += 1