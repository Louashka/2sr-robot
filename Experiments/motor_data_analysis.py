import os
import pandas as pd
import numpy as np
# from scipy.signal import butter, filtfilt
from scipy.signal import hilbert, butter, filtfilt, find_peaks
import matplotlib.pyplot as plt

def butter_lowpass_filter(data, cutoff, fs, order):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'Data/motor_pi_log.csv')

df = pd.read_csv(filename, names=["target", "voltage", "velocity", "angle", "time"])

df['time'] = df['time'] - df.iloc[0]['time']

velocity_avrg = df.loc[:, 'velocity'].mean()
print('Average amplitude: ' + str(velocity_avrg))
velocity = df['velocity'] - velocity_avrg

analytic_signal = hilbert(velocity)
amplitude_envelope = np.abs(analytic_signal)

amplitude_envelope = pd.DataFrame(amplitude_envelope).rolling(7, center=False).mean()
amplitude_envelope = amplitude_envelope.dropna().iloc[:,0].tolist()

start_i = df.shape[0] - len(amplitude_envelope)

T = df.iloc[-1]['time'] - df.iloc[start_i]['time']
# print('Sample period: ' + str(T))
n = len(amplitude_envelope)
# print('Total number of samples: ' + str(n))
fs = n / T
# print('Sampling Freq: ' + str(fs))
signal_freq = 8 / T
# print('Signal Freq: ' + str(signal_freq))

instantaneous_phase = np.unwrap(np.angle(analytic_signal))
instantaneous_frequency = (np.diff(instantaneous_phase) /
                           (2.0*np.pi) * fs)

cutoff = 2
nyq = 0.5 * fs
order = 2

# filtered_envelope = butter_lowpass_filter(amplitude_envelope, cutoff, fs, order)
amp_peaks, _ = find_peaks(amplitude_envelope)

envelope_center = np.array(amplitude_envelope).mean()
amp_avrg = np.array(amplitude_envelope)[amp_peaks].mean()
freq_avrg = np.array(instantaneous_frequency).mean() * 0.86 / 2

period = 1 / freq_avrg
print('Period: ' + str(period))

sine_wave = velocity_avrg + amp_avrg * np.sin(
    freq_avrg * df['time'].to_numpy())

peaks_time = np.array(df.iloc[start_i:]['time'])[amp_peaks]


plt.plot([df.iloc[0]['time'], df.iloc[-1]['time']], [velocity_avrg, velocity_avrg], 'k--', lw = 2, label='Avrg value')
plt.plot(df['time'], velocity + velocity_avrg, label='Measured velocity')
# plt.plot(df.iloc[start_i:]['time'], amplitude_envelope, 'k')
# plt.plot(df.iloc[start_i:]['time'], filtered_envelope)
# plt.plot(peaks_time, np.array(filtered_envelope)[amp_peaks], '.')
plt.plot(df['time'], sine_wave, 'r', lw = 2, label='Envelope')

font_size = 18

plt.xlabel('Time [s]', fontsize=font_size)
plt.ylabel('Amplitude [rad/s]', fontsize=font_size)

plt.rcParams.update({'font.size': font_size})
plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)

plt.legend()
plt.axis('equal')
plt.show()
