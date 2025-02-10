import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

blue_color = '#0E595C'
red_color = '#DC5956'
dark_grey_color = '#333333'

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['text.usetex'] = False
plt.rcParams['font.size'] = 8

def read_sm_files():
    directory = 'Experiments/Data/Tracking/Grasp'
    traverse_sm_files = [f for f in os.listdir(directory) if f.startswith('traverse_sm_') and f.endswith('.json')]
    
    traverse_data = []
    transport_data = []

    for file in traverse_sm_files:
        file_path = os.path.join(directory, file)
        with open(file_path, 'r') as f:
            file_data = json.load(f)
            traverse_data.append(file_data['traversing'])
            transport_data.append(file_data['transport'])

    return traverse_data, transport_data

def process_traverse_data(all_traverse_data):
     # Create a 3x4 subplot layout
    fig = plt.figure(figsize=(14, 8))
    
    # Create gridspec with different row heights and custom spacing
    gs = fig.add_gridspec(4, 4, 
                         width_ratios=[1, 1, 1, 1],
                         wspace=0.16,
                         bottom=0.33)
    
    for idx in range(4):        
        # Get data for this experiment
        data = all_traverse_data[idx]

        timestamps = []
        
        k1 = []
        k2 = []

        theta_errors = []

        k1_errors = []
        k2_errors = []

        temp_timestamps = []
        k1_temp = []
        k2_temp = []
        k1_stiff = []
        k2_stiff = []
    
        for data_item in list(data.values()):
            timestamps.extend(data_item['tracking']['time'])

            target_config = data_item['target_config']
            robot_states = data_item['tracking']['robot_states']
            stiffness = data_item['tracking']['target_mm']
            temp_data = data_item['tracking']['temperature']

            k1.extend(state[3] for state in robot_states)
            k2.extend(state[4] for state in robot_states)

            theta_errors .extend([(state[2]-target_config[2]) for state in robot_states])
            k1_errors.extend([abs(state[3]-target_config[3]) for state in robot_states])
            k2_errors.extend([abs(state[4]-target_config[4]) for state in robot_states])

            for i in range(len(temp_data)):
                if temp_data[i] is not None:
                    rel_time = np.array(temp_data[i]['relative_timestamps']) + data_item['tracking']['time'][i]
                    temp_timestamps.extend(rel_time.tolist())

                    k1_temp.extend([temp[0] for temp in temp_data[i]['meas']])
                    k2_temp.extend([temp[1] for temp in temp_data[i]['meas']])

            k1_stiff.extend([stiff[0] for stiff in stiffness])
            k2_stiff.extend([stiff[1] for stiff in stiffness])


        ax_theta = fig.add_subplot(gs[0, idx])
        ax_theta.plot(theta_errors, '-', color=dark_grey_color, linewidth=1)        
        # Set labels and ticks
        if idx == 0:
            ax_theta.set_ylabel(r'$\theta$ error [rad]')
        
        ax_curv = fig.add_subplot(gs[1, idx])
        ax_curv.plot(k1, '-', color=blue_color, label='seg1', linewidth=1)
        ax_curv.plot(k2, '-', color=red_color, label='seg2', linewidth=1)         
        # Set labels and ticks
        if idx == 0:
            ax_curv.set_ylabel(r'Curvature [$m^{-1}$]')
            ax_curv.legend()
        
        ax_temp = fig.add_subplot(gs[2, idx])
        ax_temp.plot(temp_timestamps, k1_temp, '-', color=blue_color, label='seg1', linewidth=1)
        ax_temp.plot(temp_timestamps, k2_temp, '-', color=red_color, label='seg2', linewidth=1)         
        # Set labels and ticks
        if idx == 0:
            ax_temp.set_ylabel('Temperature [C]')
            ax_temp.legend()

        ax_stiff = fig.add_subplot(gs[3, idx])
        ax_stiff.plot(timestamps, k1_stiff, '-', color=blue_color, label='seg1', linewidth=1)
        ax_stiff.plot(timestamps, k2_stiff, color=red_color, label='seg2', linewidth=2, linestyle='dotted')         
        # Set labels and ticks
        ax_stiff.set_xlabel('Time [s]')
        if idx == 0:
            ax_stiff.set_ylabel('Stiffness')
            ax_stiff.legend()

    plt.show()
    


all_traverse_data, all_transport_data = read_sm_files()
process_traverse_data(all_traverse_data)
