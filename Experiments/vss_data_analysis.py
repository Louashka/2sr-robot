import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

set_point = 53.5
line_width = 3
font_size = 16

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'Data/vss_log2.csv')

df = pd.read_csv(filename, names=["stiffness", "T", "mode", "time"])

df['state command'] = [0 if x == 'cooling' else 1 for x in df['mode']]
df['time'] = df['time'] - df.iloc[0]['time']

df['cycle'] = (df['mode'] != df['mode'].shift()).cumsum()
heating_rows = df[(df['mode'] == 'heating') & (df['T'] >= 38)]
grouped_heating = heating_rows.groupby('cycle')

cooling_rows = df[df['mode'] == 'cooling']
grouped_cooling = cooling_rows.groupby('cycle')

heating_time = 0
cooling_time = 0

for name, group in grouped_heating:
    heating_time += (group.iloc[-1]['time'] - group.iloc[0]['time'])

for name, group in grouped_cooling:
    cooling_time += (group.iloc[-1]['time'] - group.iloc[0]['time'])

heating_time /= 3
cooling_time /= 3

print('Heating time: ' + str(heating_time))
print('Cooling time: ' + str(cooling_time))

window = 300

temp_smoothed = df['T'].rolling(window, center=False).mean()
temp_smoothed = temp_smoothed.dropna().tolist()

plt.rcParams.update({'font.size': font_size})

fig, ax1 = plt.subplots(figsize=(15,5))

color = (0.627, 0, 0)
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('State commands', color=color)
ax1.plot(df.iloc[:-window+1]['time'], df.iloc[:-window+1]['state command'], color=color, lw=line_width)
ax1.set_ylim(-0.05, 1.095)
ax1.tick_params(axis='both', labelsize=font_size)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_yticks([0,1])

ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

color = (0, 0.44, 0.57)
ax2.set_ylabel('Temperature [C]', color=color)  # we already handled the x-label with ax1
# ax2.plot([df.iloc[0]['time'], df.iloc[-window+1]['time']], [set_point, set_point], 'k--')
ax2.plot(df.iloc[:-window+1]['time'], temp_smoothed, color=color, lw=line_width)
ax2.tick_params(axis='y', labelcolor=color, labelsize=font_size)

fig.tight_layout()  # otherwise the right y-label is slightly clipped

# plt.savefig(os.path.join(dirname, 'Figures/phase_transitions_analysis.svg'), dpi=600)
plt.show()

# plt.plot(df.iloc[:-window+1]['time'], temp_smoothed, label='Temperature')
# plt.plot([df.iloc[0]['time'], df.iloc[-window+1]['time']], [set_point, set_point], 'k--', lw = 2, label='Set point')

# plt.legend()
# plt.show()