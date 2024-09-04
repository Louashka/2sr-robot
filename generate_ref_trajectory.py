from Model import robot2sr, global_var as gv, splines
from Controller import robot2sr_controller

import numpy as np
import matplotlib.pyplot as plt
from gekko import GEKKO

K_MAX = np.pi / gv.L_VSS
cardiod = splines.Cardioid(1)

def arc(theta0, k, seg=1):
    l = np.linspace(0, gv.L_VSS, 50)
    flag = -1 if seg == 1 else 1
    theta_array = theta0 + flag * k * l

    if k == 0:
        x = np.array([0, flag * gv.L_VSS * np.cos(theta0)])
        y = np.array([0, flag * gv.L_VSS * np.sin(theta0)])
    else:
        x = np.sin(theta_array) / k - np.sin(theta0) / k
        y = -np.cos(theta_array) / k + np.cos(theta0) / k

    theta = theta_array[-1]
        
    return [x, y, theta % (2 * np.pi)]

def plot(q_start, q_target, original_traj, ref_traj):
    # Plot the initial robot's configuration
    vss1_init = arc(q_start[2] + q_start[0], q_start[3] + q_start[1], seg=1)
    vss2_init = arc(q_start[2] + q_start[0], q_start[4] + q_start[1], seg=2)

    plt.plot(vss1_init[0], vss1_init[1], 'r-', alpha=0.5)
    plt.plot(vss2_init[0], vss2_init[1], 'b-', alpha=0.5)

    # Plot the target robot's configuration
    vss1_target = arc(q_target[2], q_target[3], seg=1)
    vss2_target = arc(q_target[2], q_target[4], seg=2)

    plt.plot(vss1_target[0] + q_target[0], vss1_target[1] + q_target[1], 'r-')
    plt.plot(vss2_target[0] + q_target[0], vss2_target[1] + q_target[1], 'b-')

    # Plot the original trajectory
    plt.plot(original_traj[0], original_traj[1], 'k.')

    # Plot the reference trajectory
    plt.plot(ref_traj[0], ref_traj[1], 'm.')

    plt.tight_layout()
    plt.axis('equal')
    plt.show()


def calc_original_traj(q_start, stiff, v):
    agent = robot2sr.Robot(1, *q_start, stiffness=stiff)
    agent_controller = robot2sr_controller.Controller()

    agent_controller.update_agent(agent, agent.config)

    v_fk = [0, 0, 0] + v

    original_traj_x = [agent.x]
    original_traj_y = [agent.y]

    timer = 0

    while timer < 20:
        _, q = agent_controller.move(agent, v_fk, agent.stiffness)
        agent_controller.update_agent(agent, q)

        original_traj_x.append(q[0])
        original_traj_y.append(q[1])

        timer += 1

        if np.abs(q[3]) >= K_MAX or np.abs(q[4]) >= K_MAX:
            print('break')
            break

    return [original_traj_x, original_traj_y], agent


def estimate_ref_trajectory(q_start, q_target, stiff, n):
    if stiff == [1, 0]:
        seg = 1
        lu = 2

    if stiff == [0, 1]:
        seg = 2
        lu = 1

    if stiff == [1, 1]:
        pass

    k_array = np.linspace(q_start[2+seg], q_target[2+seg], n)
    theta_array = np.linspace(q_start[2], q_target[2], n)

    dk_dt = np.average(np.diff(k_array))
    dtheta_dt = np.average(np.diff(theta_array))
    dk_dt_ = dtheta_dt / gv.L_VSS

    # print(dk_dt / dk_dt_)
    print(dk_dt_ - dk_dt_ / dk_dt)
    print(dtheta_dt)

    k_interp = []
    for i in range(n):
        k_interp.append(q_start[2+seg] + dk_dt_ * i)

    ref_traj = []

    for k, k_, theta in zip(k_array, k_interp, theta_array):
        k_res = k_
        ref_pos = cardiod.pos(k_res, lu)

        rot_theta = theta - (-1)**lu * k_res * gv.L_VSS
        # rot_theta = 0

        rot = np.array([[np.cos(rot_theta), -np.sin(rot_theta)],
                        [np.sin(rot_theta), np.cos(rot_theta)]])
        
        ref_pos = rot @ np.array(ref_pos)
        
        ref_traj.append(ref_pos.tolist())

    ref_traj  = np.array(ref_traj).T

    offset = [q_start[0] - ref_traj[0, 0], q_start[1] - ref_traj[1, 0]]
    ref_traj += np.array(offset).reshape(2, 1)

    ref_traj_x = ref_traj[0,:].tolist()
    ref_traj_y = ref_traj[1,:].tolist()

    return [ref_traj_x, ref_traj_y]

def mpc_controller(k_start, k_target, n_steps):
    cardioid1 = splines.Cardioid(1)
    cardioid2 = splines.Cardioid(2)

    m = GEKKO(remote=False)
    
    # Time horizon
    m.time = np.linspace(0, 1, n_steps)
    
    # Parameters
    k_start = m.Param(value=[k_start] * n_steps)  # Initialize as array
    k_target = m.Param(value=[k_target] * n_steps)  # Initialize as array

    a = m.Param(value=gv.CARDIOID_A[0])
    phi_max = m.Param(value=gv.CARDIOID_TH_MAX)
    phi_min = m.Param(value=gv.CARDIOID_TH_MIN)
    l = m.Param(value=gv.L_VSS)

    var_phi = 2 * np.pi / (l * (phi_max - phi_min))
    
    # Variables
    v2 = m.MV(value=0, lb=-0.2, ub=0.2)  # Adjust bounds as necessary
    k = m.CV()
    
    # Initial conditions
    k.value = k_start
    k.STATUS = 1
    
    # Control variables
    v2.STATUS = 1
    
    # Governing equations from splines (to be filled in)
    m.Equation(k.dt() == var_phi / (2 * a * (1 - m.cos(phi_min + (1 / var_phi) * (k + np.pi / l)))) * v2)
    
    # Objective: Minimize the difference between q and q_target
    m.Obj((k - k_target)**2)
    
    # Solver options
    m.options.IMODE = 6  # MPC mode
    m.options.CV_TYPE = 1  # Control variable type
    
    # Solve
    m.solve(disp=False)
    
    # Extract optimized velocities
    v2_opt = v2.value
    
    return v2_opt

if __name__ == "__main__":
    q_start = [0, 0, 0, 0.0, 0]

    stiff = [1, 0]
    v = [-0.03, 0]

    # stiff = [0, 1]
    # v = [0.04, 0.0]

    original_traj, agent = calc_original_traj(q_start, stiff, v)
    # ref_traj = estimate_ref_trajectory(q_start, agent.config, stiff, len(original_traj[0]))

    # plot(q_start, agent.config, original_traj, ref_traj)

    v_predicted = mpc_controller(q_start[3], agent.k1, len(original_traj[0]))
    print(v_predicted)
