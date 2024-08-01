import matplotlib.pyplot as plt
from scipy.optimize import minimize
import cvxpy
import math
import numpy as np
import cubic_spline_planner
from Model import splines, robot2sr, global_var as gv
from Controller import robot2sr_controller

show_animation = True
NX = 3  # x = x, y, yaw
NU = 2  # a = [linear velocity, linear velocity, angular velocity ]
NW = 4  # number of wheels
T = 10  # horizon length

# mpc parameters
R = np.diag([10000, .001])  # input cost matrix
Q = np.diag([10, 10, 0.0])  # agent cost matrix
Qf = Q # agent final matrix
Rd = np.diag([10000, 0.01])

GOAL_DIS = 2  # goal distance
STOP_SPEED = 0.02   # stop speed
MAX_TIME = 200.0  # max simulation time

# iterative paramter
MAX_ITER = 1  # Max iteration
DU_TH = 0.1  # iteration finish param

N_IND_SEARCH = 10  # Search index number
DT = gv.DT  # [s] time tick

TARGET_SPEED = 0.08   # [m/s] target speed

MAX_SPEED = 15  # Maximum speed in rad/s
MIN_SPEED = 0.5  # Minimum non-zero speed in rad/s 

agent_controller = robot2sr_controller.Controller()

def get_linear_model_matrix(vref, phi):
    A = np.zeros((NX, NX))
    A[0, 0] = 1.0
    A[0, 2] = -vref * math.cos(phi)*DT
    A[1, 1] = 1.0
    A[1, 2] = -vref * math.sin(phi)*DT
    A[2, 2] = 1.0

    B = np.zeros((NX, NU))
    B[0, 0] = -DT * math.sin(phi)
    B[0, 1] = 0 #0
    B[1, 0] = DT * math.cos(phi)
    B[1, 1] = 0 #0
    B[2, 1] = DT

    return A, B

# def get_linear_model_matrix(v_ref, theta_ref):
#     A = np.zeros((NX, NX))
#     A[0, 0] = 1.0
#     A[0, 2] = -(v_ref[0]  * math.sin(theta_ref) + v_ref[1]  * math.cos(theta_ref)) * DT
#     A[1, 1] = 1.0
#     A[1, 2] = (v_ref[0]  * math.cos(theta_ref) - v_ref[1]  * math.sin(theta_ref)) * DT
#     A[2, 2] = 1.0

#     B = np.zeros((NX, NU))
#     B[0, 0] = DT * math.cos(theta_ref)
#     B[0, 1] = -DT * math.sin(theta_ref)
#     B[1, 0] = DT * math.sin(theta_ref)
#     B[1, 1] = DT * math.cos(theta_ref)
#     B[2, 2] = DT
 
#     return A, B

def get_nparray_from_matrix(x):
    return np.array(x).flatten()

def plot_arrow(pose, length=0.05, width=0.1, fc="r", ec="k"):
    yaw = pose[2] + np.pi / 2
    plt.arrow(pose[0], pose[1], length * math.cos(yaw), length * math.sin(yaw),
        fc=fc, ec=ec, head_width=width, head_length=width)
    plt.plot(pose[0], pose[1])

def calc_nearest_index(agent, cx, cy, cyaw, pind):
    dx = [agent.x - icx for icx in cx[pind:(pind + N_IND_SEARCH)]]
    dy = [agent.y - icy for icy in cy[pind:(pind + N_IND_SEARCH)]]
    d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]
    mind = min(d)
    ind = d.index(mind) + pind
    mind = math.sqrt(mind)
    dxl = cx[ind] - agent.x
    dyl = cy[ind] - agent.y
    angle = cyaw[ind] - math.atan2(dyl, dxl)
    if angle < 0:
        mind *= -1
    return ind, mind


def calc_ref_trajectory(cx, cy, cyaw, sp, dl, ind, cur_vel):
    xref = np.zeros((NX, T + 1))
    vref = np.zeros((1, T + 1))
    ncourse = len(cx)

    xref[0, 0] = cx[ind]
    xref[1, 0] = cy[ind]
    xref[2, 0] = cyaw[ind]
    vref[0, 0] = sp[ind]
    travel = 0.0
    for i in range(1, T + 1):
        travel += abs(cur_vel) * DT
        dind = int(round(travel / dl))
        if (ind + dind) < ncourse:
            xref[0, i] = cx[ind + dind]
            xref[1, i] = cy[ind + dind]
            xref[2, i] = cyaw[ind + dind]
            vref[0, i] = sp[ind + dind]
        else:
            xref[0, i] = cx[ncourse - 1]
            xref[1, i] = cy[ncourse - 1]
            xref[2, i] = cyaw[ncourse - 1]
            vref[0, i] = sp[ncourse - 1]

    return xref, vref

def check_goal(agent, goal, tind, nind,vi):
    dx = agent.x - goal[0]
    dy = agent.y - goal[1]
    d = math.hypot(dx, dy)
    isgoal = (d <= GOAL_DIS)
    if abs(tind - nind) >= 5:
        isgoal = False
    isstop = (abs(vi) <= STOP_SPEED)
    if isgoal and isstop:
        return True
    return False

def predict_motion(agent_config: list, shape: tuple, v_list: list, omega_list: list):
    qbar = np.zeros(shape)
    qbar[:, 0] = agent_config[:3]

    agent = robot2sr.Robot(1, *agent_config)
    update_agent(agent, agent.config)
    for (i, v, omega) in zip(range(1, T + 1), v_list, omega_list):
        q = agent_controller.get_config(agent, [0, v, omega, 0, 0], agent.stiffness)
        update_agent(agent, q)
        qbar[:, i] = agent.pose

    return qbar


def linear_mpc_control(xref, xbar, x0, vref, wheels):    
    x = cvxpy.Variable((NX, T + 1))
    u = cvxpy.Variable((NU, T))
    vw = cvxpy.Variable((NW, T)) #to calculate vr and vl

    # Add binary variables for each wheel at each time step
    z = cvxpy.Variable((NW, T), boolean=True)
 
    cost = 0.0
    constraints = []

    for t in range(T):
        cost += cvxpy.quad_form(u[:, t], R)
        if t != 0:
            cost += cvxpy.quad_form(x[:, t], Q)        
        A, B = get_linear_model_matrix(vref[0, t], xbar[2, t])  

        constraints += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t]]  

        for w in range(NW):
            constraints += [vw[w, t] == (1 / gv.WHEEL_R) * (math.sin(wheels[w][2]) * (u[0, t] + vref[0, t+1]) + 
                                                           (wheels[w][0] * math.sin(wheels[w][2]) - wheels[w][1] * math.cos(wheels[w][2])) * u[1, t])]

        if t < (T - 1):
            cost += cvxpy.quad_form((u[:, t + 1] - u[:, t]), Rd)

    # constraints += [vw[:,:] <= 15]  
    # constraints += [vw[:,:] >= -15]    

    for w in range(NW):
        # Exclude velocities from -1 to 1
        constraints += [vw[w, :] >= MIN_SPEED - (MAX_SPEED + MIN_SPEED) * z[w, :]]
        constraints += [vw[w, :] <= -MIN_SPEED + (MAX_SPEED + MIN_SPEED) * (1 - z[w, :])]

    cost += cvxpy.quad_form(x[:, T], Qf)
    constraints += [x[:, 0] ==  x0 - xref[:,0]]    
    prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
    prob.solve(solver=cvxpy.ECOS_BB, verbose=False)
    #OSQP,CVXOPT, ECOS, scs


    if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
        x_new = get_nparray_from_matrix(x.value[0, :])
        y_new = get_nparray_from_matrix(x.value[1, :])
        theta_new = get_nparray_from_matrix(x.value[2, :])
        v_new = get_nparray_from_matrix(u.value[0, :])
        omega_new = get_nparray_from_matrix(u.value[1, :])
        
        wheels_velocity = np.array(vw.value)
        
        x_new += xref[0,:]
        y_new += xref[1,:]
        theta_new += xref[2,:]
        v_new += vref[0, 1:]

    else:
        print("Error: Cannot solve mpc..")
        v_new, omega_new, x_new, y_new, theta_new = None, None, None, None, None

    return v_new, omega_new, x_new, y_new, theta_new, wheels_velocity

def arc(agent: robot2sr.Robot, seg=1) -> tuple[np.ndarray, np.ndarray, float]:
    k = agent.curvature[seg-1]
    l = np.linspace(0, gv.L_VSS, 50)
    flag = -1 if seg == 1 else 1
    theta_array = agent.theta + flag * k * l

    if k == 0:
        x = np.array([0, flag * gv.L_VSS * np.cos(agent.theta)])
        y = np.array([0, flag * gv.L_VSS * np.sin(agent.theta)])
    else:
        x = np.sin(theta_array) / k - np.sin(agent.theta) / k
        y = -np.cos(theta_array) / k + np.cos(agent.theta) / k

    theta_end = theta_array[-1]
        
    return x, y, theta_end % (2 * np.pi)

def update_agent(agent: robot2sr.Robot, q: np.ndarray):
    agent.config = q

    vss1 = arc(agent)
    vss1_conn_x = [agent.x + vss1[0][-1] - gv.L_CONN * np.cos(vss1[2]), agent.x + vss1[0][-1]]
    vss1_conn_y = [agent.y + vss1[1][-1] - gv.L_CONN * np.sin(vss1[2]), agent.y + vss1[1][-1]]

    lu_head_x = vss1_conn_x[0] + np.sqrt(2) / 2 * gv.LU_SIDE * np.cos(vss1[2] + np.pi + np.pi / 4)
    lu_head_y = vss1_conn_y[0] + np.sqrt(2) / 2 * gv.LU_SIDE * np.sin(vss1[2] + np.pi + np.pi / 4)

    agent.head.pose = [lu_head_x, lu_head_y, vss1[2]]


    vss2 = arc(agent, 2)
    vss2_conn_x = [agent.x + vss2[0][-1], agent.x + vss2[0][-1] + gv.L_CONN * np.cos(vss2[2])]
    vss2_conn_y = [agent.y + vss2[1][-1], agent.y + vss2[1][-1] + gv.L_CONN * np.sin(vss2[2])]

    lu_tail_x = vss2_conn_x[1] + np.sqrt(2) / 2 * gv.LU_SIDE * np.cos(vss2[2] - np.pi / 4)
    lu_tail_y = vss2_conn_y[1] + np.sqrt(2) / 2 * gv.LU_SIDE * np.sin(vss2[2] - np.pi / 4)

    agent.tail.pose = [lu_tail_x, lu_tail_y, vss2[2]] 


if __name__ == '__main__':
    agent = robot2sr.Robot(1, 0, 0, -np.pi/2, 0, 0)
    update_agent(agent, np.array([0, 0, -np.pi/2, 0, 0]))
    wheels, q = agent_controller.move(agent, [0] * 5, agent.stiffness)
    
    dl = 0.01 # course tick
    # ax = [0.0, 1.0, 1.5, 3.0, 4.0]
    # ay = [0.0, 1.0, 1.5, 1.0, 0.0] 
    # cx, cy, cyaw, _, s = cubic_spline_planner.calc_spline_course(ax, ay, ds=dl)

    path_x = np.arange(0, 2, 0.01)
    path_y = np.array([np.sin(x / 0.21) * x / 15.0 for x in path_x])
    path = splines.Trajectory(path_x, path_y)

    cx, cy, cyaw, s = path.params

    sp = []
    for i in range(1, len(cx)):
        sp.append(TARGET_SPEED * (1 - path.curvature[i]/(max(path.curvature)+5)))
    sp.append(0)

    goal = [cx[-1], cy[-1]]

    time = 0.0
    x, y, yaw, v, omega, t = [agent.x], [agent.y], [agent.theta], [0.0], [0.0], [0.0]

    vi, omegai = 0,0
    ox, oy, oyaw = None, None, None
    ovr, ovl = None, None 

    ov = [0.0] * T
    oomega = [0.0] * T

while MAX_TIME >= time:
        target_ind = path.getTarget(agent.position, dl)
        xref, vref = calc_ref_trajectory(cx, cy, cyaw, sp, dl, target_ind, vi)     

        xbar = predict_motion(agent.config, xref.shape, [vi] * T, [omegai] * T)
        ov, oomega, ox, oy, oyaw, wheels_v = linear_mpc_control(xref, xbar, agent.pose, vref, wheels)
        # print(wheels_v[:,0])

        if oomega is not None:
            vi , omegai = ov[0], oomega[0]

        wheels, q = agent_controller.move(agent, [0, vi, omegai, 0, 0], agent.stiffness)
        update_agent(agent, q)
        time = time + DT
            
        x.append(agent.x)
        y.append(agent.y)
        yaw.append(agent.theta)
        v.append(vi)
        t.append(time)
        omega.append(omegai)

        if check_goal(agent, goal, target_ind, len(cx), vi):
            print("Goal")
            break       
        if target_ind % 1 == 0 and show_animation == True:
            plt.cla()
            plot_arrow(agent.pose)
            plt.plot(cx, cy, "-r", label="course")
            plt.plot(x, y, "-b", label="trajectory")
            plt.plot(xbar[0,:], xbar[1,:], '-k', label='ref')
            plt.plot(cx[target_ind], cy[target_ind], "xg", label="target")
            plt.axis("equal")
            plt.grid(True)
            plt.title("speed[m/sec]:" + str(round(vi, 2)))
            plt.pause(0.0005)

if show_animation:  # pragma: no cover
    plt.close("all")
    plt.subplots()
    plt.plot(cx, cy, "-r", label="spline")
    plt.plot(x, y, "-g", label="tracking")
    plt.grid(True)
    plt.axis("equal")
    plt.xlabel("x[m]")
    plt.ylabel("y[m]")
    plt.legend()
    plt.subplots()
    plt.plot(t, v, "-r", label="speed")
    plt.grid(True)
    plt.xlabel("Time [s]")
    plt.ylabel("Speed [kmh]")
    plt.show()