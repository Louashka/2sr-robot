import json
import glob
import os
import re
import numpy as np
import seaborn as sns
import pandas as pd
from scipy import signal, stats
from scipy.spatial.distance import euclidean
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from brokenaxes import brokenaxes


json_files = glob.glob('Experiments/Data/Tracking/Grasping/Old/grasp_exp_*.json')
loaded_data = {}
pattern = re.compile(r'grasp_exp_(\w+)_(\d+)')

for file_name in json_files:
    match = pattern.search(file_name)
    if match:
        object_type = match.group(1)  # e.g., "bean"
        object_number = int(match.group(2))  # e.g., 1
        
        # Initialize nested dictionary if this object type hasn't been seen yet
        if object_type not in loaded_data:
            loaded_data[object_type] = {}
        
        # Load the JSON data
        with open(file_name, 'r') as file:
            loaded_data[object_type][object_number] = json.load(file)

# Create an empty list to store individual DataFrames
all_pregrasp_dfs = []

window_size = 9

# Iterate through all object types and numbers
for object_type in loaded_data:
    for object_number in loaded_data[object_type]:
        try:
            # Extract approach data for the current object
            pregrasp_data = {}
            n = len(loaded_data[object_type][object_number]['grasp']['1']['tracking']['time'])
            
            # Add type and number columns
            pregrasp_data['type'] = [object_type] * n
            pregrasp_data['number'] = [object_number] * n
            
            # Extract tracking data
            current_grasp = loaded_data[object_type][object_number]['grasp']['1']['tracking']
            
            # Time data
            ts= (np.array(current_grasp['time']) - current_grasp['time'][0]).tolist()
            for i in range(len(ts)):
                if ts[i] > 60:
                    ts[i] -= 35

            pregrasp_data['timestamps'] = ts
            
            # Robot states
            robot_states = np.array(current_grasp['robot']['states'])
            if n > 12:
                pregrasp_data['robot_x'] = robot_states[:, 0].tolist()
                pregrasp_data['robot_y'] = savgol_filter(robot_states[:, 1].tolist(), window_size, 2)
                pregrasp_data['robot_th'] = savgol_filter(robot_states[:, 2].tolist(), window_size, 2)

                pregrasp_data['k1'] = savgol_filter(robot_states[:, 3].tolist(), window_size, 2)
                pregrasp_data['k2'] = savgol_filter(robot_states[:, 4].tolist(), window_size, 2)
                
                # Robot velocities
                robot_vel = np.array(current_grasp['robot']['target_vel'])
                pregrasp_data['v_x'] = robot_vel[:, 0].tolist()
                pregrasp_data['v_y'] = robot_vel[:, 1].tolist()
                pregrasp_data['omega'] = robot_vel[:, 2].tolist()
                pregrasp_data['v_1'] = robot_vel[:, 3].tolist()
                pregrasp_data['v_2'] = robot_vel[:, 4].tolist()

            # Robot transitions
            pregrasp_data['transitions'] = current_grasp['robot']['transitions']

            # Object states
            object_states = np.array(current_grasp['object']['states'])
            pregrasp_data['object_x'] = object_states[:, 0].tolist()
            pregrasp_data['object_y'] = object_states[:, 1].tolist()
            pregrasp_data['object_th'] = object_states[:, 2].tolist()
            
            # Create DataFrame for this object
            current_df = pd.DataFrame(pregrasp_data)
            
            # Append to the list of DataFrames
            all_pregrasp_dfs.append(current_df)
            
        except KeyError as e:
            print(f"Skipping {object_type} {object_number} due to missing data: {e}")
        except IndexError as e:
            print(f"Skipping {object_type} {object_number} due to index error: {e}")

# Combine all DataFrames into one
combined_pregrasp_df = pd.concat(all_pregrasp_dfs, ignore_index=True)


current_df = combined_pregrasp_df.loc[(combined_pregrasp_df['type'] == 'heart') & 
                                     (combined_pregrasp_df['number'] == 1)]

sns.set_theme(style='whitegrid', palette='muted', font='sans-serif', font_scale=1.7)

xlims=((38.720204000128431, 40.862157900119201), (86.329274500021711, 91.529274500021711))
line_w = 3

# Create figure with 2x2 broken axes subplots
fig = plt.figure(figsize=(12, 10))
sps1, sps2, sps3, sps4, sps5, sps6 = GridSpec(3,2, hspace=0.25, wspace=0.28)

bax1 = brokenaxes(xlims=xlims, fig=fig, subplot_spec=sps1, d=0.006, diag_color='m')
bax1.plot(current_df['timestamps'], 10 * current_df['robot_x'], label='x', lw=line_w)
bax1.plot(current_df['timestamps'], 10 * current_df['robot_y'], 'r', label='y', lw=line_w)
bax1.set_ylim(-1.6, 1.2)
bax1.set_ylabel(r'Pos [m] $\times 10^{-1}$')
bax1.legend(loc='center left')

bax1.spines['top'][0].set_visible(True)
bax1.spines['top'][1].set_visible(True)
bax1.spines['right'][0].set_visible(True)


bax2 = brokenaxes(xlims=xlims, fig=fig, subplot_spec=sps3, d=0.006, diag_color='m')
bax2.plot(current_df['timestamps'], current_df['robot_th'], 'k', lw=line_w)
bax2.set_ylim(0.40, 1.2)
bax2.set_ylabel(r'$\theta$ [rad]', labelpad=35)

bax2.spines['top'][0].set_visible(True)
bax2.spines['top'][1].set_visible(True)
bax2.spines['right'][0].set_visible(True)


bax3 = brokenaxes(xlims=xlims, fig=fig, subplot_spec=sps5, d=0.006, diag_color='m')
bax3.plot(current_df['timestamps'], current_df['k1'], label='VSS1', lw=line_w)
bax3.plot(current_df['timestamps'], current_df['k2'], 'r', label='VSS2', lw=line_w)
# bax3.set_ylim(-1.6, 1.2)
bax3.set_ylabel(r'Curvature [m$^{-1}$]')
bax3.legend(loc='lower right')
bax3.set_xlabel('Time [s]', labelpad=32)

bax3.spines['top'][0].set_visible(True)
bax3.spines['top'][1].set_visible(True)
bax3.spines['right'][0].set_visible(True)


ax4 = plt.Subplot(fig, sps2)
ax4.plot(current_df['timestamps'][4:], 100 * current_df['v_x'][4:], lw=line_w, label=r'$v_x$')
ax4.plot(current_df['timestamps'][4:], 100 * current_df['v_y'][4:], lw=line_w, label=r'$v_y$', color='r')
ax4.legend()
ax4.set_ylabel('Lin vel [cm/s]')

fig.add_subplot(ax4)

ax5 = plt.Subplot(fig, sps4)
ax5.plot(current_df['timestamps'][4:], 10 * current_df['omega'][4:], 'k', lw=line_w)
ax5.set_ylabel(r'$\omega$ [rad/s] $\times 10^{-1}$')

fig.add_subplot(ax5)


ax6 = plt.Subplot(fig, sps6)
ax6.plot(current_df['timestamps'][:5], 100 * current_df['v_1'][:5], lw=line_w, label=r'$v_1$')
ax6.plot(current_df['timestamps'][:5], 100 * current_df['v_2'][:5], lw=line_w, label=r'$v_2$', color='r')
ax6.legend(loc='lower right')
ax6.set_ylabel('\"Flex\" vel [cm/s]')
ax6.set_xlabel('Time [s]')

fig.add_subplot(ax6)


# plt.show()

plt.savefig('Experiments/Figures/pregrasp_system_response.pdf', format='pdf', dpi=150, bbox_inches='tight')

