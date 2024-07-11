import numpy as np
import matplotlib.pyplot as plt

traj_x = np.arange(0, 2, 0.01)
traj_y = [np.sin(x / 0.21) * x / 2.7 for x in traj_x]

print(abs(min(traj_y)) + max(traj_y))

plt.plot(traj_x, traj_y)
plt.axis('equal')
plt.show()