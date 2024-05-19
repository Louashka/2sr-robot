import numpy as np
import matplotlib.pyplot as plt

T_d = 64

t1 = 5.6
t2 = t1 + 30
t3 = t2 + 33

n = 200
line_w = 10

font_size = 40

fig, ax = plt.subplots(figsize=(14, 7))

heating_line = [np.linspace(0, t1, n), np.linspace(0, T_d, n)]
t_sin = np.linspace(t1, t2, 500)
sine_wave = heating_line[1][-1] + 3 * np.sin(1.5 * t_sin - 2)
cooling_line = [np.linspace(t2, t3, n), np.linspace(sine_wave[-1], 0, n)]

x = np.concatenate([heating_line[0], t_sin, cooling_line[0]])
y = np.concatenate([heating_line[1], sine_wave, cooling_line[1]])
z = y / 10

graph = ax.scatter(x, y, c=z, lw = line_w, cmap=plt.cm.coolwarm, marker='.')
cb = plt.colorbar(graph)
cb.set_label('Voltage [V]', size=font_size)
cb.ax.tick_params(labelsize=font_size)


ax.set_xlabel('Time [s]', fontsize=font_size)
ax.set_ylabel('Temperature [C]', fontsize=font_size)

plt.rcParams.update({'font.size': font_size})
ax.tick_params(axis='x', labelsize=font_size)
ax.tick_params(axis='y', labelsize=font_size)

# plt.axis('equal')
plt.show()