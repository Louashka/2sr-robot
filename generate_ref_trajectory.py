from Model import robot2sr, global_var, splines
from Controller import robot2sr_controller

import numpy as np
import matplotlib.pyplot as plt

agent_controller = robot2sr_controller.Controller()

k_max = np.pi / (2 * global_var.L_VSS)

q_start = [0, 0, 0, -2, 0]
agent = robot2sr.Robot(1, *q_start, stiffness=[1, 0])
agent_controller.update_agent(agent, agent.config)

v2 = 0.02
v_fk = [0, 0, 0, 0, v2]

original_traj_array = []
target_configs = []
original_k_array = []

counter = 0

while counter < 8:
    timer = 0
    original_traj_x = [agent.x]
    original_traj_y = [agent.y]
    original_k = [agent.k1]

    while timer < 10:
        _, q = agent_controller.move(agent, v_fk, agent.stiffness)
        agent_controller.update_agent(agent, q)

        original_traj_x.append(q[0])
        original_traj_y.append(q[1])
        original_k.append(agent.k1)
        timer += 1

        if np.abs(q[3]) >= k_max or np.abs(q[4]) >= k_max:
            print('break')
            break

    original_traj_array.append([original_traj_x, original_traj_y])

    target_configs.append(agent.config) 
    agent_controller.update_agent(agent, q_start)
    original_k_array.append(original_k)

    counter += 1
    v2 += 0.025
    v_fk = [0, 0, 0, 0, v2]


#------------------------------------------------------------------------------
# Generate reference trajectory

ref_traj_array = []
ref_k_array = []

for i in range(len(original_traj_array)):
    target_config = target_configs[i]
    ref_traj = []

    N = len(original_traj_array[i][0])
    k_array = np.linspace(q_start[3], target_config[3], N)
    ref_k_array.append(k_array)
    spiral1 = splines.LogSpiral(1)

    for k in k_array:
        ref_pos = spiral1.get_pos(k)

        ref_traj.append(ref_pos)

    offset_th = 0
    rot_theta = q_start[2] - offset_th
    rot = np.array([[np.cos(rot_theta), -np.sin(rot_theta)],
                    [np.sin(rot_theta), np.cos(rot_theta)]])

    ref_traj = rot @ np.array(ref_traj).T

    offset = [q_start[0] - ref_traj[0, 0], q_start[1] - ref_traj[1, 0]]
    ref_traj += np.array(offset).reshape(2, 1)
    

    ref_traj_x = ref_traj[0,:]
    ref_traj_y = ref_traj[1,:]

    ref_traj_array.append([ref_traj_x, ref_traj_y])


fig, axs = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle('Original vs Reference Trajectories')

for i in range(len(original_traj_array)):
    row = i // 4
    col = i % 4
    
    original_traj_x, original_traj_y = original_traj_array[i]
    ref_traj_x, ref_traj_y = ref_traj_array[i]
    
    axs[row, col].plot(original_traj_x, original_traj_y, 'b.', label='Original')
    axs[row, col].plot(original_traj_x, original_traj_y, 'b-')
    
    axs[row, col].plot(ref_traj_x, ref_traj_y, 'r.', label='Reference')
    axs[row, col].plot(ref_traj_x, ref_traj_y, 'r-')

    axs[row, col].set_aspect('equal')
    axs[row, col].legend()
    axs[row, col].set_title(f'Iteration {i+1}')
    axs[row, col].set_xlabel('X')
    axs[row, col].set_ylabel('Y')

plt.tight_layout()
plt.show()
