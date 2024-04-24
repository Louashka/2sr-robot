import numpy as np
import matplotlib.pyplot as plt

T_d = 64

t1 = 5.6
t2 = t1 + 30
t3 = t2 + 33

n = 200

font_size = 20

heating_line = [np.linspace(0, t1, n), np.linspace(0, T_d, n)]
t_sin = np.linspace(t1, t2, 500)
sine_wave = T_d + 3 * np.sin(3 * t_sin + 2)
cooling_line = [np.linspace(t2, t3, n), np.linspace(sine_wave[-1], 0, n)]

x = np.concatenate([heating_line[0], t_sin, cooling_line[0]])
y = np.concatenate([heating_line[1], sine_wave, cooling_line[1]])
z = y / 10

graph = plt.scatter(x, y, c=z, lw = 3, cmap=plt.cm.coolwarm, marker='.')
cb = plt.colorbar(graph)
cb.set_label('Voltage [V]', size=font_size)
cb.ax.tick_params(labelsize=font_size)


plt.xlabel('Time [s]', fontsize=font_size)
plt.ylabel('Temperature [C]', fontsize=font_size)

plt.rcParams.update({'font.size': font_size})
plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)

plt.axis('equal')
plt.show()