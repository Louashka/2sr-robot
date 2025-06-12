import os
import json
import numpy as np
import pandas as pd
import scipy.fftpack as sf
import scipy.signal as sig
from scipy.fft import rfft, rfftfreq
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

names = ['c', 'e', 'h', 'b']
names_boxplot = ['c', 'e', 'h', 'b', 'h-g']
trial_idx = [2, 1, 2, 0]
img_idx = [2, 2, 3, 2]


# sns.set_style("whitegrid", {'axes.grid' : False})

# plt.rcParams['pdf.fonttype'] = 42
# plt.rcParams['text.usetex'] = False
# plt.rcParams['font.size'] = 20

sns.set_theme(style='whitegrid', palette='muted', font='sans-serif', font_scale=1.9)

colors = ['#d5bb67', 'k', '#d66060', '#4779cf']
line_width = 2.5  # Common variable for line width

window_size = 11


def transport_files():
    directory = 'Experiments/Data/Tracking/Transport'

    cheescake_files = [f for f in os.listdir(directory) if f.startswith('cheescake') and f.endswith('.json')]
    ellipse_files = [f for f in os.listdir(directory) if f.startswith('ellipse') and f.endswith('.json')]
    heart_files = [f for f in os.listdir(directory) if f.startswith('heart') and f.endswith('.json')]
    bean_files = [f for f in os.listdir(directory) if f.startswith('bean') and f.endswith('.json')]
    
    cheescake_data = []
    ellipse_data = []
    heart_data = []
    bean_data = []

    for file in cheescake_files:
        file_path = os.path.join(directory, file)
        with open(file_path, 'r') as f:
            file_data = json.load(f)
            cheescake_data.append([file_data['path'], file_data['tracking']])

    for file in ellipse_files:
        file_path = os.path.join(directory, file)
        with open(file_path, 'r') as f:
            file_data = json.load(f)
            ellipse_data.append([file_data['path'], file_data['tracking']])

    for file in heart_files:
        file_path = os.path.join(directory, file)
        with open(file_path, 'r') as f:
            file_data = json.load(f)
            heart_data.append([file_data['path'], file_data['tracking']])

    for file in bean_files:
        file_path = os.path.join(directory, file)
        with open(file_path, 'r') as f:
            file_data = json.load(f)
            bean_data.append([file_data['path'], file_data['tracking']])

    return cheescake_data, ellipse_data, heart_data, bean_data

def filter(x):
    x = np.array(x)
    Fs = 200

    fc = np.array([5,15])
    wc = 2 * fc / Fs
    [b, a] = sig.butter(3, 0.1, btype='low')

    x_filt = sig.lfilter(b, a, x) 


    return x_filt


def calc_tracking_error(entry):
    path_data = entry[0]
    tracking_data = entry[1]

    # Process reference path
    path = []
    for path_point in path_data:
        path.append([path_point['x'], path_point['y'], path_point['yaw']])
    path = np.array(path)[:, :2]  # Only take x,y coordinates

    # Process tracking data
    timestamps = []
    object_traj = []
    robot_traj = []

    for item in tracking_data:
        timestamps.append(item['time'])
        object_traj.append(item['object']['pose'])
        robot_traj.append(item['robot']['pose'])
    
    # Convert to numpy arrays
    timestamps = np.array(timestamps[5:])
    object_traj = np.array(object_traj[5:])  # Only take x,y coordinates

    object_traj_x = savgol_filter(object_traj[:, 0], window_size, 2)
    object_traj_y = savgol_filter(object_traj[:, 1], window_size, 2)

    
    # Calculate tracking error at each timestep
    errors = []
    for obj_pos in zip(object_traj_x, object_traj_y):
        # Calculate distances to all points in the path
        distances = np.linalg.norm(path - obj_pos, axis=1)
        # Get minimum distance
        min_distance = np.min(distances)
        errors.append(1000 * min_distance)

    # smoothed_errors = savgol_filter(errors, window_size, 2)

    # return timestamps, smoothed_errors
    
    return timestamps, errors, filter(errors)
    

def plot_pos(idx):
    ellipse_entry = data[idx][trial_idx[idx]]
    tracking_data = ellipse_entry[1]

    # Process tracking data
    timestamps = []
    object_traj = []
    robot_traj = []

    for item in tracking_data:
        timestamps.append(item['time'])
        object_traj.append(item['object']['pose'])
        robot_traj.append(item['robot']['pose'])
    
    # Convert to numpy arrays
    timestamps = np.array(timestamps)
    object_traj = np.array(object_traj)

    return timestamps, object_traj[:,0], object_traj[:,1]

def process_transport_data():    
    fig, axs = plt.subplots(4, 2, figsize=(13, 15))

    circle_idx = 0
    circle_entry = data[1][circle_idx]


    path_data = circle_entry[0]
    tracking_data = circle_entry[1]

    # Process tracking data
    timestamps = []
    object_vel = []
    robot_vel = []
    object_traj = []
    robot_traj = []

    for item in tracking_data:
        timestamps.append(item['time'])
        object_vel.append(item['object']['target_velocity'])
        robot_vel.append(item['robot']['target_velocity'])
        object_traj.append(item['object']['pose'])
        robot_traj.append(item['robot']['pose'])
    
    # Convert to numpy arrays
    timestamps = np.array(timestamps)
    object_vel = np.array(object_vel)
    robot_vel = np.array(robot_vel)

    robot_vel_h = savgol_filter(robot_vel[:,1], window_size, 1)
    robot_vel_l = savgol_filter(robot_vel[:,0], window_size, 1)

    object_vel_h = savgol_filter(object_vel[:,1], window_size, 1)
    object_vel_l = savgol_filter(object_vel[:,0], window_size, 1)

    robot_omega = savgol_filter(robot_vel[:,2], window_size, 1)
    object_omega = savgol_filter(object_vel[:,2], window_size, 1)

    axs[0, 0].plot(timestamps, 100 * robot_vel_h, label='robot', color=colors[1], linewidth=line_width)
    axs[0, 0].plot(timestamps, 100 * object_vel_l, label='object', color='b', linewidth=line_width)

    axs[0, 0].set_ylabel(r'$v_h$ [cm/s]') 
    axs[0, 0].legend()
    axs[0, 0].set_title('Ellipse 3')

    axs[1, 0].plot(timestamps, 100 * robot_vel_l, label='robot', color=colors[1], linewidth=line_width)
    axs[1, 0].plot(timestamps, -100 * object_vel_h, label='object', color='b', linestyle='--', linewidth=line_width)

    axs[1, 0].set_ylabel(r'$v_l$ [cm/s]') 
    axs[1, 0].legend(ncol=2)

    axs[2, 0].plot(timestamps, 100 * robot_omega, label='robot', color=colors[1], linewidth=line_width)
    axs[2, 0].plot(timestamps, 100 * object_omega, label='object', color='b', linewidth=line_width)
    axs[2, 0].legend(ncol=2)

    axs[2, 0].set_xlabel('Time [s]')
    axs[2, 0].set_ylabel(r'$\omega$ [rad/s] $\times 10^{-2}$')  

    object_traj = np.array(object_traj)
    robot_traj = np.array(robot_traj)

    robot_x = savgol_filter(robot_traj[:,0], window_size, 1)
    robot_y = savgol_filter(robot_traj[:,1], window_size, 1)
    robot_th = savgol_filter(robot_traj[:,2], window_size, 1)

    object_x = savgol_filter(object_traj[:,0], window_size, 1)
    object_y = savgol_filter(object_traj[:,1], window_size, 1)
    object_th = savgol_filter(object_traj[:,2], window_size, 1)

    # Process reference path
    path = []
    for path_point in path_data:
        path.append([path_point['x'], path_point['y'], path_point['yaw']])
    path = np.array(path)[:, :2]  # Only take x,y coordinates
    
    axs[3, 0].plot(10 * robot_x, 10 * robot_y, label='robot path', color=colors[1], linewidth=line_width)
    axs[3, 0].plot(10 * object_x, 10 * object_y, label='object path', color='b', linewidth=line_width)
    axs[3, 0].plot(10 * path[:,0], 10 * path[:,1], label='target path', color=colors[2], linestyle='dotted', linewidth=line_width)

    axs[3, 0].set_xlabel(r'x [m] $\times 10^{-1}$') 
    axs[3, 0].set_ylabel(r'y [m] $\times 10^{-1}$') 
    axs[3, 0].legend()


    axs[0, 1].plot(timestamps, robot_th, label='robot', color=colors[1], linewidth=line_width)
    axs[0, 1].plot(timestamps, object_th, label='object', color='b', linewidth=line_width)

    axs[0, 1].set_ylabel(r'$\theta$ [rad]') 
    axs[0, 1].legend()
    axs[0, 1].set_title('Ellipse 3')


    heart_tracking_data = heart_data[0][1]

    heart_timestamps = []
    heart_object_theta = []
    heart_robot_theta = []

    for item in heart_tracking_data:
        heart_timestamps.append(item['time'])
        heart_object_theta.append(item['object']['pose'][2])
        heart_robot_theta.append(item['robot']['pose'][2])

    heart_robot_theta = savgol_filter(heart_robot_theta, window_size, 1)
    heart_object_theta = savgol_filter(heart_object_theta, window_size, 1)

    axs[1, 1].plot(heart_timestamps, heart_robot_theta, label='robot', color=colors[1], linewidth=line_width)
    axs[1, 1].plot(heart_timestamps, heart_object_theta, label='object', color='b', linewidth=line_width)

    axs[1, 1].set_ylabel(r'$\theta$ [rad]') 
    axs[1, 1].legend()
    axs[1, 1].set_title('Heart 2')

    
    bean_tracking_data = bean_data[1][1]

    bean_timestamps = []
    bean_object_theta = []
    bean_robot_theta = []

    for item in bean_tracking_data:
        bean_timestamps.append(item['time'])
        bean_object_theta.append(item['object']['pose'][2])
        bean_robot_theta.append(item['robot']['pose'][2])

    bean_robot_theta = savgol_filter(bean_robot_theta, window_size, 1)
    bean_object_theta = savgol_filter(bean_object_theta, window_size, 1)

    axs[2, 1].plot(bean_timestamps, bean_robot_theta, label='robot', color=colors[1], linewidth=line_width)
    axs[2, 1].plot(bean_timestamps, bean_object_theta, label='object', color='b', linewidth=line_width)

    axs[2, 1].set_xlabel('Time [s]')
    axs[2, 1].set_ylabel(r'$\theta$ [rad]') 
    axs[2, 1].legend()
    axs[2, 1].set_title('Bean 1')


    # First subplot
    for shape_data, name, trial, img_id, color in zip(data, names, trial_idx, img_idx, colors):
        entry = shape_data[trial]
        tracking_data = entry[1]

        # Process tracking data
        current_timestamps = []
        current_object_traj = []
        current_robot_traj = []

        for item in tracking_data:
            current_timestamps.append(item['time'])
            current_object_traj.append(item['object']['pose'])
            current_robot_traj.append(item['robot']['pose'])
        
        # Convert to numpy arrays
        current_timestamps = np.array(current_timestamps)
        current_object_traj = np.array(current_object_traj)

        # Apply moving average to smooth the error data
        
        current_smoothed_theta = savgol_filter(current_object_traj[:,2], window_size, 1)

        # Normalize timestamps to range [0, 1]
        normalized_timestamps = (current_timestamps - current_timestamps[0]) / (current_timestamps[-1] - current_timestamps[0])
        object_traj = np.array(object_traj)
    
        axs[3, 1].plot(normalized_timestamps, current_smoothed_theta, label=f'{name} {img_id}', color=color, linewidth=line_width)

    axs[3, 1].set_ylabel(r'$\theta$ [rad]')
    axs[3, 1].legend(loc='upper center', ncol=2)
    axs[3, 1].set_xlabel('Normalized Time')

    
    # Adjust spacing between subplots
    fig.tight_layout()
    # plt.show()
    plt.savefig('Experiments/Figures/transport_path_plots.pdf', format='pdf', dpi=150, bbox_inches='tight', transparent=True)



def plot_errors():
    trial_idx = [1, 2, 0, 2]
    img_idx = [1, 1, 2, 3]  

    fig, axs = plt.subplots(2, 2, figsize=(9.2, 9))

    gs = axs[0, 0].get_gridspec()
    for ax in axs[0, :]:
        ax.remove()
    axbig = fig.add_subplot(gs[0, :])

    for shape_data, name, trial, img_id, color in zip(data, names, trial_idx, img_idx, colors):
        entry = shape_data[trial]

        timestamps, errors, smoothed_errors = calc_tracking_error(entry)

        normalized_timestamps = (timestamps - timestamps[0]) / (timestamps[-1] - timestamps[0])

        axbig.plot(normalized_timestamps, smoothed_errors, color=color, linewidth=2)

        mean_error = np.mean(smoothed_errors)

        axbig.plot(normalized_timestamps, [mean_error]*len(normalized_timestamps), label=f'{name} {img_id} mean', color=color, linestyle='--', linewidth=2)


        timestamps_ = np.array(timestamps) 
        timestamps_ -= timestamps_[0]

        samples = np.where(timestamps_ > 1)[0][0] + 1
        sampling_rate = 1/samples

        # Remove mean and trend for better frequency analysis
        detrended_data = sig.detrend(smoothed_errors)
        
        # Apply window to reduce spectral leakage
        windowed_data = detrended_data * sig.windows.hann(len(detrended_data))
        
        # Compute FFT
        n = len(windowed_data)
        fft_result = rfft(windowed_data)
        
        # Compute frequency bins
        frequencies = rfftfreq(n, 1/sampling_rate)
        
        # Compute magnitude spectrum (normalize by n/2 for proper scaling)
        amplitudes = np.abs(fft_result) * 2 / n
        
        # Identify dominant frequencies (peaks in the spectrum)
        # Exclude very low frequencies (< 0.1 Hz) often related to trends
        peaks, _ = sig.find_peaks(amplitudes[frequencies > 0.001], height=0.05*np.max(amplitudes))
        
        # Get the actual frequencies and amplitudes of these peaks
        dominant_freqs = []
        for peak in peaks:
            actual_idx = peak + np.sum(frequencies <= 0.001)  # Adjust index for excluded low freqs
            dominant_freqs.append((frequencies[actual_idx], amplitudes[actual_idx]))
        
        # Sort by amplitude (descending)
        dominant_freqs.sort(key=lambda x: x[1], reverse=True)

        axs[1, 0].plot(frequencies, amplitudes, label=f'{name} {img_id}', color=color, linewidth=2)

        # for freq, amp in dominant_freqs[:3]:  # Show top 3
        #     axs[1].plot(freq, amp, 'ro')
        #     axs[1].text(freq, amp, f' {freq:.2f} Hz', verticalalignment='bottom')

    axbig.legend(ncol=2)
    axbig.set_xlabel('Normalized Time')
    axbig.set_ylabel('e [mm]')
    axbig.set_title('Tracking Error')

    axs[1, 0].set_xlabel('Frequency (Hz)')
    axs[1, 0].set_ylabel('Amplitude')
    axs[1, 0].set_title('Frequency Spectrum')
    axs[1, 0].set_xlim(-0.001, 0.025)
    axs[1, 0].legend()


    all_shapes = []
    all_trials = []
    all_errors = []
    all_errors_median = [] 
    all_errors_std = [] 

    idx_order = {'circle': [3, 1, 2], 'ellipse': [3, 2, 1], 'heart': [2, 1, 3], 'bean': [2, 1, 3]}

    for shape_data, name, color in zip(data, idx_order.keys(), colors):
        for trial in range(len(shape_data)):
            _, _, smoothed_errors = calc_tracking_error(shape_data[trial])

            all_shapes.append(name)
            all_trials.append(idx_order[name][trial])
            all_errors.append(smoothed_errors)
            all_errors_median.append(np.median(smoothed_errors))
            all_errors_std.append(np.std(smoothed_errors))


    errors_dict = {'shape': all_shapes, 'Trial': all_trials, 'Error': all_errors, 'Median e [mm]': all_errors_median, 'std': all_errors_std}
    errors_df = pd.DataFrame(data=errors_dict)

    for shape, color in zip(errors_df['shape'].unique(), colors):
        shape_data = errors_df[errors_df['shape'] == shape]
        shape_data = shape_data.sort_values(by='Trial')
        axs[1, 1].errorbar(x='Trial', y='Median e [mm]', data=shape_data, label=shape, color=color)
        axs[1, 1].scatter(shape_data['Trial'], shape_data['Median e [mm]'], color=color)

        # axs[1, 1].boxplot(shape_data['Error'], tick_labels=[1, 2, 3])

    axs[1, 1].legend(ncol=2)
    axs[1, 1].set_xlabel('Trial')
    axs[1, 1].set_ylabel('e [mm]')
    axs[1, 1].set_title('Median Error')
    
    fig.tight_layout()
    # plt.show()

    plt.savefig('Experiments/Figures/transport_errors.pdf', format='pdf', dpi=150, bbox_inches='tight', transparent=True)



cheescake_data, ellipse_data, heart_data, bean_data = transport_files()
data = [cheescake_data, ellipse_data, heart_data, bean_data]

process_transport_data()
# plot_errors()