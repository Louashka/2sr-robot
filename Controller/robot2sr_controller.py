import serial
import math
import numpy as np
from typing import List
from cvxopt import matrix, solvers
from Model import global_var, robot2sr, splines
import cvxpy
from gekko import GEKKO
from threading import Thread
import time

port_name = "COM3"
serial_port = serial.Serial(port_name, 115200)


class Controller:
    def __init__(self) -> None:
        self.max_linear_velocity = 0.2  # Maximum desired linear velocity
        self.max_angular_velocity = 1.0  # Maximum desired angular velocity
        self.lookahead_distance = 0.15  # Adjust based on your robot's size and desired responsiveness
        self.kp_linear = 3  # Proportional gain for linear velocity
        self.kp_angular = 1  # Proportional gain for angular velocity

        self.dt = global_var.DT

        self.sp = None

        self.NX = 5  # x = x, y, yaw
        self.NU = 2  # a = [linear velocity, linear velocity, angular velocity ]
        self.NW = 4  # number of wheels
        self.T = 11  # horizon length

        # mpc parameters
        self.R = np.diag([1, 10000, 0.0015])  # input cost matrix
        self.Q = np.diag([10, 10, 0.0])  # agent cost matrix
        self.Qf = self.Q # agent final matrix
        self.Rd = np.diag([10, 10000, 0.001])

        self.TARGET_SPEED = 0.08
        self.MAX_SPEED = 15  # Maximum speed in rad/s
        self.MIN_SPEED = 0.5  # Minimum non-zero speed in rad/s 

        self.sc = StiffnessController()

        self.cardioid1 = splines.Cardioid(1)
        self.cardioid2 = splines.Cardioid(2)

        self.gekko_solver()

    def motionPlannerMPC(self, agent: robot2sr.Robot, path: splines.Trajectory, v_current) -> tuple[List[float], List[float]]:        
        cx, cy, cyaw, s = path.params

        if self.sp is None:
            self.sp = []
            for i in range(1, len(cx)):
                self.sp.append(self.TARGET_SPEED * (1 - path.curvature[i]/(max(path.curvature) + 5)))
            self.sp.append(0)

        # r_rot_cost = 0.002 - (0.002 - 0.0008) * path.curvature[path.last_idx] / max(path.curvature)
        # self.R = np.diag([10, 10000, r_rot_cost])

        target_ind = path.getTarget(agent.position, self.lookahead_distance)
        qref, vref = self.__calc_ref_trajectory(cx, cy, cyaw, self.sp, target_ind, v_current[0]) 
        # qbar = self.__predict_motion(agent.pose, qref.shape, [0, v_current[0], v_current[1], 0, 0], [0, 0])

        head_wheels, _ = self._calcWheelsCoords(agent.pose, agent.head.pose)
        tail_wheels, _ = self._calcWheelsCoords(agent.pose, agent.tail.pose, lu_type='tail')
        wheels = head_wheels + tail_wheels

        vx_predicted, vy_predicted, omega_predicted = self.__linear_mpc_control(qref, agent.pose, vref, wheels)

        return [vx_predicted[0], vy_predicted[0], omega_predicted[0], 0, 0], agent.stiffness
    
    def __calc_ref_trajectory(self, cx: list, cy: list, cyaw: list, sp, ind, v) -> tuple[np.ndarray, np.ndarray]:
        qref = np.zeros((self.NX, self.T + 1))
        vref = np.zeros((1, self.T + 1))
        ncourse = len(cx)

        qref[0, 0] = cx[ind]
        qref[1, 0] = cy[ind]
        qref[2, 0] = cyaw[ind]
        vref[0, 0] = sp[ind]
        travel = 0.0

        for i in range(1, self.T + 1):
            travel += abs(v) * self.dt
            dind = int(round(travel / self.lookahead_distance))
            if (ind + dind) < ncourse:
                qref[0, i] = cx[ind + dind]
                qref[1, i] = cy[ind + dind]
                qref[2, i] = cyaw[ind + dind]
                vref[0, i] = sp[ind + dind]
            else:
                qref[0, i] = cx[ncourse - 1]
                qref[1, i] = cy[ncourse - 1]
                qref[2, i] = cyaw[ncourse - 1]
                vref[0, i] = sp[ncourse - 1]

        return qref, vref

    
    def update_agent(self, agent: robot2sr.Robot, q: np.ndarray):
        agent.config = q

        vss1 = self.__arc(agent)
        vss1_conn_x = [agent.x + vss1[0][-1] - global_var.L_CONN * np.cos(vss1[2]), agent.x + vss1[0][-1]]
        vss1_conn_y = [agent.y + vss1[1][-1] - global_var.L_CONN * np.sin(vss1[2]), agent.y + vss1[1][-1]]

        lu_head_x = vss1_conn_x[0] + np.sqrt(2) / 2 * global_var.LU_SIDE * np.cos(vss1[2] + np.pi + np.pi / 4)
        lu_head_y = vss1_conn_y[0] + np.sqrt(2) / 2 * global_var.LU_SIDE * np.sin(vss1[2] + np.pi + np.pi / 4)

        agent.head.pose = [lu_head_x, lu_head_y, vss1[2]]

        vss2 = self.__arc(agent, 2)
        vss2_conn_x = [agent.x + vss2[0][-1], agent.x + vss2[0][-1] + global_var.L_CONN * np.cos(vss2[2])]
        vss2_conn_y = [agent.y + vss2[1][-1], agent.y + vss2[1][-1] + global_var.L_CONN * np.sin(vss2[2])]

        lu_tail_x = vss2_conn_x[1] + np.sqrt(2) / 2 * global_var.LU_SIDE * np.cos(vss2[2] - np.pi / 4)
        lu_tail_y = vss2_conn_y[1] + np.sqrt(2) / 2 * global_var.LU_SIDE * np.sin(vss2[2] - np.pi / 4)

        agent.tail.pose = [lu_tail_x, lu_tail_y, vss2[2]]

    def __arc(self, agent: robot2sr.Robot, seg=1) -> tuple[np.ndarray, np.ndarray, float]:
        k = agent.curvature[seg-1]
        l = np.linspace(0, global_var.L_VSS, 50)
        flag = -1 if seg == 1 else 1
        theta_array = agent.theta + flag * k * l

        if k == 0:
            x = np.array([0, flag * global_var.L_VSS * np.cos(agent.theta)])
            y = np.array([0, flag * global_var.L_VSS * np.sin(agent.theta)])
        else:
            x = np.sin(theta_array) / k - np.sin(agent.theta) / k
            y = -np.cos(theta_array) / k + np.cos(agent.theta) / k

        theta_end = theta_array[-1]
            
        return x, y, theta_end % (2 * np.pi)
    
    def __linear_mpc_control(self, qref: np.ndarray, agent_pose: list, vref: np.ndarray, wheels: list):
        q = cvxpy.Variable((self.NX, self.T + 1))
        u = cvxpy.Variable((self.NU, self.T))
        vw = cvxpy.Variable((self.NW, self.T)) #to calculate vr and vl

        # Add binary variables for each wheel at each time step
        z = cvxpy.Variable((self.NW, self.T), boolean=True)
    
        cost = 0.0
        constraints = []

        constraints += [q[:, 0] == agent_pose - qref[:,0]]  

        for t in range(self.T):
            cost += cvxpy.quad_form(u[:, t], self.R)
            if t != 0:
                cost += cvxpy.quad_form(q[:, t], self.Q)        
            A, B = self.__get_linear_model_matrix(vref[0, t], qref[2, t])  

            constraints += [q[:, t + 1] == A @ q[:, t] + B @ u[:, t]]  

            for w in range(self.NW):
                constraints += [vw[w, t] == (1 / global_var.WHEEL_R) * (math.cos(wheels[w][2]) * u[0, t] + 
                                                            math.sin(wheels[w][2]) * (u[1, t] + vref[0, t+1]) + 
                                                            (wheels[w][0] * math.sin(wheels[w][2]) - wheels[w][1] * math.cos(wheels[w][2])) * u[2, t])]

            # if t < (self.T - 1):
            #     cost += cvxpy.quad_form((u[:, t + 1] - u[:, t]), self.Rd)


        for w in range(self.NW):
            # Exclude velocities from -1 to 1
            constraints += [vw[w, :] >= self.MIN_SPEED - (self.MAX_SPEED + self.MIN_SPEED) * z[w, :]]
            constraints += [vw[w, :] <= -self.MIN_SPEED + (self.MAX_SPEED + self.MIN_SPEED) * (1 - z[w, :])]

        cost += cvxpy.quad_form(q[:, self.T], self.Qf)  
        prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
        prob.solve(solver=cvxpy.ECOS_BB, verbose=False)

        if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
            vx_predicted = np.array(u.value[0, :]).flatten()
            vy_predicted = np.array(u.value[1, :]).flatten() + vref[0, 1:]
            omega_predicted = np.array(u.value[2, :]).flatten()
        else:
            print("Error: Cannot solve mpc..")
            vx_predicted, vy_predicted, omega_predicted = None, None, None

        return vx_predicted, vy_predicted, omega_predicted
    
    def __get_linear_model_matrix(self, vref, phi):
        A = np.zeros((self.NX, self.NX))
        A[0, 0] = 1.0
        A[0, 2] = -vref * math.cos(phi) * self.dt
        A[1, 1] = 1.0
        A[1, 2] = -vref * math.sin(phi) * self.dt
        A[2, 2] = 1.0

        B = np.zeros((self.NX, self.NU))
        B[0, 0] = self.dt * math.cos(phi)
        B[0, 1] = -self.dt * math.sin(phi)
        B[1, 0] = self.dt * math.sin(phi)
        B[1, 1] = self.dt * math.cos(phi)
        B[2, 2] = self.dt

        return A, B
    
    
    def mhe(self, ref_traj: splines.Trajectory, k_array: np.ndarray):
        n = len(ref_traj.x)

        m = GEKKO(remote=False)
        m.time = np.linspace(0, global_var.DT * (n-1), n)

        # x = m.CV(value=ref_traj.x)
        # y = m.CV(value=ref_traj.y)
        # theta = m.CV(value=ref_traj.yaw)
        k1 = m.CV(value=k_array)

        # x.FSTATUS = 1
        # y.FSTATUS = 1
        # theta.FSTATUS = 1
        k1.FSTATUS = 1

        u2 = m.MV(value=0.0, lb = -0.2, ub=0.2)

        u2.STATUS = 1

        w2 = m.Intermediate(-(1 / global_var.WHEEL_R) * u2)
        w2_curve = m.Intermediate(w2**4 - self.MIN_SPEED * w2**2)

        # k1_ratio = self.spiral2.get_k_dot(k1.VALUE) / self.spiral1.get_k_dot(k1.VALUE)
        # pos_lu2 = self.spiral1.get_pos_dot(theta.VALUE, k1.VALUE, 1, 2)

        # Equations
        # m.Equation(x.dt() == k1_ratio * pos_lu2[0] * u2)
        # m.Equation(y.dt() == k1_ratio * pos_lu2[1] * u2)
        # m.Equation(theta.dt() == self.spiral2.get_th_dot(k1.VALUE) * u2)
        m.Equation(k1.dt() == self.spiral2.m / (self.spiral2.a * m.exp(-self.spiral2.b * k1 * global_var.L_VSS / self.spiral2.m - self.spiral2.phi0) * global_var.L_VSS) * u2)

        # Constraints
        m.Equation(w2 >= -self.MAX_SPEED)
        m.Equation(w2 <= self.MAX_SPEED)
        m.Equation(w2_curve >= 0)

        m.options.IMODE = 5
        m.options.EV_TYPE = 2
        m.solve(disp=False)

        return [0] * n, u2.VALUE
    
    def _calc_soft_ref_traj(self, config_traj: np.ndarray, target_ind: int, n: int) -> np.ndarray:
        q_ref = np.zeros((self.NX, self.T))

        for i in range(0, self.T):
            if target_ind + i < n:
                q_ref[0, i] = config_traj[0, target_ind + i]
                q_ref[1, i] = config_traj[1, target_ind + i]
                q_ref[2, i] = config_traj[2, target_ind + i]
                q_ref[3, i] = config_traj[3, target_ind + i]
                q_ref[4, i] = config_traj[4, target_ind + i]
            else:
                q_ref[0, i] = config_traj[0, -1]
                q_ref[1, i] = config_traj[1, -1]
                q_ref[2, i] = config_traj[2, -1]
                q_ref[3, i] = config_traj[3, -1]
                q_ref[4, i] = config_traj[4, -1]

        return q_ref
    
    def gekko_solver(self) -> None:
        self.m = GEKKO(remote=False)
        self.m.time = np.linspace(0, global_var.DT * (self.T-1), self.T)

        # Manipulated variables        
        self.u1 = self.m.MV(value=0.00, lb=-0.2, ub=0.2)
        self.u1.STATUS = 1

        self.u2 = self.m.MV(value=0.00, lb=-0.2, ub=0.2)
        self.u2.STATUS = 1
        # self.u2.DCOST = 10

        # State variables
        self.x = self.m.SV(value=[0.0] * self.T)
        self.y = self.m.SV(value=0.0)
        self.theta = self.m.SV(value=0.0)
        self.k1 = self.m.SV(value=0.0)

        # # Additional variables
        # self.phi1 = self.m.Var()
        # self.rho1 = self.m.Var()

        # self.phi2 = self.m.Var()
        # self.rho2 = self.m.Var()

        # self.spiral1_k1_dot = self.m.Var()
        # self.spiral2_k1_dot = self.m.Var()
        # self.coef = self.m.Var()

        # self.th = self.m.Var()
        # self.x_dot = self.m.Var()
        # self.y_dot = self.m.Var()

        # Define an intermediate variable for wheel speed
        w1 = self.m.Intermediate(-(1 / global_var.WHEEL_R) * self.u1)
        w1_curve = self.m.Intermediate(w1**4 - self.MIN_SPEED * w1**2)

        w2 = self.m.Intermediate(-(1 / global_var.WHEEL_R) * self.u2)
        w2_curve = self.m.Intermediate(w2**4 - self.MIN_SPEED * w2**2)

        # Equations
        # self.m.Equation(self.phi1 == self.cardioid1.phi_min + (1 / self.cardioid1.var_phi) * (self.k1 + np.pi / global_var.L_VSS))
        # self.m.Equation(self.rho1 == 2 * self.cardioid1.a * (1 - self.m.cos(self.phi1)))
        # self.m.Equation(self.phi2 == self.cardioid2.phi_min + (1 / self.cardioid2.var_phi) * (self.k1 + np.pi / global_var.L_VSS))
        # self.m.Equation(self.rho2 == 2 * self.cardioid2.a * (1 - self.m.cos(self.phi2)))
        # self.m.Equation(self.spiral1_k1_dot == self.cardioid1.var_phi / self.rho1)
        # self.m.Equation(self.spiral2_k1_dot == self.cardioid2.var_phi / self.rho2)
        # self.m.Equation(self.coef == self.spiral2_k1_dot / self.spiral1_k1_dot)
        # self.m.Equation(self.th == self.theta - self.k1 * global_var.L_VSS)
        # self.m.Equation(self.x_dot == self.rho1 * (self.m.sin(self.phi1) - self.m.sin(2 * self.phi1)))
        # self.m.Equation(self.y_dot == self.rho1 * (-self.m.cos(self.phi1) + self.m.cos(2 * self.phi1)))
        
        
        # self.m.Equation(self.x.dt() == self.coef * (self.m.cos(self.th) * self.x_dot - self.m.sin(self.th) * self.y_dot) * self.u2)
        # self.m.Equation(self.y.dt() == self.coef * (self.m.sin(self.th) * self.x_dot + self.m.cos(self.th) * self.y_dot) * self.u2)
        # self.m.Equation(self.theta.dt() == (global_var.L_VSS * self.spiral2_k1_dot) * self.u2)
        # self.m.Equation(self.k1.dt() == -self.spiral1_k1_dot * self.u1 + self.spiral2_k1_dot * self.u2)
        k2_ratio = self.cardioid2.k_dot(self.k1.VALUE) / self.cardioid1.k_dot(self.k1.VALUE)
        pos_lu2 = self.cardioid1.pos_dot(self.theta.VALUE, self.k1.VALUE, 1, 2)

        self.m.Equation(self.x.dt() == k2_ratio * pos_lu2[0] * self.u2)
        self.m.Equation(self.y.dt() == k2_ratio * pos_lu2[1] * self.u2)
        self.m.Equation(self.theta.dt() == self.cardioid2.th_dot(self.k1.VALUE) * self.u2)
        self.m.Equation(self.k1.dt() == -self.cardioid1.k_dot(self.k1.VALUE) * self.u1 + self.cardioid2.k_dot(self.k1.VALUE) * self.u2)
        
        # Constraints
        self.m.Equation(w1 >= -self.MAX_SPEED)
        self.m.Equation(w1 <= self.MAX_SPEED)
        self.m.Equation(w1_curve >= 0)

        self.m.Equation(w2 >= -self.MAX_SPEED)
        self.m.Equation(w2 <= self.MAX_SPEED)
        self.m.Equation(w2_curve >= 0)

        # Options
        self.m.options.IMODE = 6  # MPC mode
        self.m.options.SOLVER = 1

    
    def softMPC(self, agent: robot2sr.Robot, path: splines.TrajectoryShape, config_traj: np.ndarray, v_target: list) -> tuple[float, float]:
        target_ind = path.getTarget(agent.config.tolist(), self.lookahead_distance)
        q_ref = self._calc_soft_ref_traj(config_traj, target_ind, path.n) 

        self.u1.VALUE = v_target[0]
        self.u2.VALUE = v_target[1]

        self.x.VALUE = [agent.x] * self.T
        self.y.VALUE = [agent.y] * self.T
        self.theta.VALUE = [agent.theta] * self.T
        self.k1.VALUE = [agent.k1] * self.T

        x_ref = q_ref[0, :].tolist()
        y_ref = q_ref[1, :].tolist()
        theta_ref = q_ref[2, :].tolist()
        k_ref = q_ref[3, :].tolist()

        # Objective
        cost = 0
        for i in range(self.T):
            cost += 10 * (self.x[i] - x_ref[i])**2
            cost += 10 * (self.y[i] - y_ref[i])**2
            cost += (self.theta[i] - theta_ref[i])**2
            cost += 0.001 * (self.k1[i] - k_ref[i])**2

        self.m.Obj(cost)

        # Solve
        self.m.solve(disp=False)
        
        return [self.u1.NEWVAL, self.u2.NEWVAL]


    def motionPlanner(self, agent: robot2sr.Robot, path: splines.Trajectory, states: dict) -> tuple[List[float], List[float]]:
        flag = False

        target_point = path.getTargetPoint(agent.position, self.lookahead_distance)
        desired_heading = np.arctan2(target_point[1] - agent.y, target_point[0] - agent.x) - np.pi/2
        desired_heading %= (2 * np.pi)

        target_config = target_point + [desired_heading] + agent.curvature

        for alt_idx in states:
            if int(alt_idx) - path.last_idx > 0 and int(alt_idx) - path.last_idx < 3:
                # if np.all(np.abs((states[alt_idx][3:] - np.array(agent.curvature))) < 0.5):
                #     continue

                target_config = states[alt_idx] 
                flag = True
                break

        distance = target_config[:2] - np.array(agent.position)
        heading_error = self.minAngleDistance(target_config[2], agent.theta)
        curvature_diff = target_config[3:] - np.array(agent.curvature)

        q_tilda = np.concatenate([
            self.kp_linear * distance,
            [self.kp_angular * heading_error],
            curvature_diff
        ])

        # A set of possible stiffness configurations
        s = [[0, 0], [1, 0], [0, 1], [1, 1]]
        # A set of possible configurations
        q_ = [None] * len(s)
        v_ = [None] * len(s)

        for i in range(len(s)):
            J = agent.jacobian(s[i])
            v_[i] = np.linalg.pinv(J) @ q_tilda
            q_dot = J @ v_[i]
            q_[i] = agent.config + q_dot * global_var.DT

        # Determine the stiffness configuration that promotes
        # faster approach to the target
        dist_ = np.linalg.norm(q_ - np.array(target_config), axis=1)
        min_i = np.argmin(dist_)

        # The extent of the configuration change
        delta_q_ = np.linalg.norm(agent.config - np.array(q_), axis=1)
        current_i = s.index(agent.stiffness)

        # # Stiffness transition is committed only if the previous
        # # stiffness configuration does not promote further motion
        # if min_i != current_i:
        #     if delta_q_[current_i] > 10**(-17):
        #         min_i = current_i

        if not flag:
            min_i = 0

        # v_optimal = np.linalg.pinv(agent.jacobian([0, 0])) @ q_tilda
        v_optimal = v_[min_i]

        # Apply constraints to v_optimal
        # v_constrained = self.applyCelocity_constraints(agent, v_optimal)
        
        # v = v_constrained.tolist()
        
        return v_optimal.tolist(), s[min_i]
    
    def inverse_k(self, agent: robot2sr.Robot, target_config: list) -> None:
        q_t = np.array(target_config)
        q_tilda = 2 * (q_t - agent.config) * global_var.DT

        J = agent.jacobian([0, 0])
        v = np.linalg.pinv(J) @ q_tilda

        return v, [0, 0]
    
    @staticmethod
    def minAngleDistance(initial_angle, target_angle):
        # Calculate the clockwise and counterclockwise distances
        clockwise_distance = (target_angle - initial_angle) % (2 * np.pi)
        counterclockwise_distance = (initial_angle - target_angle) % (2 * np.pi)
        
        if clockwise_distance < counterclockwise_distance:
            return -clockwise_distance
        
        return counterclockwise_distance
    
    def get_config(self, agent: robot2sr.Robot, v: List[float], s: List[float]) -> np.ndarray:
        q_dot = agent.jacobian(s) @ v
        q = agent.config + q_dot * self.dt
        
        return q
    
    def getWheelsVelocities(self, agent: robot2sr.Robot, v: List[float], s: List[float]) -> tuple[np.ndarray, List[List[float]], np.ndarray]:
        head_wheels, head_wheels_global = self._calcWheelsCoords(agent.pose, agent.head.pose)
        tail_wheels, tail_wheels_global = self._calcWheelsCoords(agent.pose, agent.tail.pose, lu_type='tail')
        wheels = head_wheels + tail_wheels
        wheels_global = head_wheels_global + tail_wheels_global

        V = self._wheelsConfigMatrix(wheels, s)
        omega = np.round(V @ v, 3)
        v_new = np.linalg.pinv(V) @ omega

        # print(omega)
        return omega, wheels, self.get_config(agent, v_new, s)

    def move(self, agent: robot2sr.Robot, v: List[float], s: List[float]) -> tuple[List[List[float]], List[float]]:
        omega, wheels, q = self.getWheelsVelocities(agent, v, s)
        commands = omega.tolist() + s + [agent.id]

        sc_feedback = [[], []]
        sc_feedback = self.sc.control_loop(agent.stiffness, s, agent.id)

        sendCommands(commands)

        return wheels, q, self.sc.states, sc_feedback

    def stop(self, agent: robot2sr.Robot) -> None:
        commands = [0, 0, 0, 0] + agent.stiffness + [agent.id]
        self._sendCommands(commands)

    @staticmethod
    def _calcWheelsCoords(agent_pose: List[float], lu_pose: List[float], lu_type='head') -> tuple[List[List[float]], List[List[float]]]:
        w1_0 = np.array([[-0.0275], [0]]) if lu_type == 'head' else np.array([[0.0275], [0]])
        w2_0 = np.array([[0.0105], [-0.0275]]) if lu_type == 'head' else np.array([[-0.0105], [-0.027]])

        R = np.array([[np.cos(lu_pose[2]), -np.sin(lu_pose[2])],
                      [np.sin(lu_pose[2]), np.cos(lu_pose[2])]])
        w1 = (R @ w1_0).T[0] + lu_pose[:2]
        w2 = (R @ w2_0).T[0] + lu_pose[:2]

        w = Controller._wheelsToBodyFrame(agent_pose, [w1, w2], lu_pose[-1], lu_type)
        return w, [w1, w2]

    @staticmethod
    def _wheelsToBodyFrame(agent_pose: List[float], w: List[List[float]], lu_theta: float, lu_type='head') -> List[List[float]]:
        R_ob = np.array([[np.cos(agent_pose[2]), -np.sin(agent_pose[2])],
                         [np.sin(agent_pose[2]), np.cos(agent_pose[2])]])
        T_ob = np.block([[R_ob, np.array([agent_pose[:2]]).T], [np.zeros((1,2)), 1]])
        T_bo = np.linalg.inv(T_ob)

        offset = 0 if lu_type == 'head' else 2
        for i in range(2):
            w_b0 = [*w[i], 1]
            w[i] = np.matmul(T_bo, w_b0)[:2]
            w[i] = np.append(w[i], (lu_theta - agent_pose[2]) % (2 * np.pi) + global_var.BETA[i+offset])
        return w
    
    @staticmethod
    def _wheelsConfigMatrix(wheels: List[List[float]], s: List[bool]) -> np.ndarray:
        flag_rigid = int(not any(s))
        flag_soft = int(any(s))

        V = np.zeros((4, 5))
        for i, w in enumerate(wheels):
            tau = w[0] * np.sin(w[2]) - w[1] * np.cos(w[2])
            V[i, :] = [flag_rigid * np.cos(w[2]), flag_rigid * np.sin(w[2]), flag_rigid * tau,
                       -flag_soft * (i == 0), -flag_soft * (i == 2)]

        return 1 / global_var.WHEEL_R * V

def sendCommands(commands: List[float]) -> None:
    msg = "s" + "".join(f"{command}\n" for command in commands)
    serial_port.write(msg.encode())


class StiffnessController:
    def __init__(self):
        self.states = [0, 0]  # Initial state

        self.liquid_threshold = 63
        self.solid_threshold = 53

        self.temp = [0, 0]

        self.send_cmd = True

    def control_loop(self, current_states: list, target_states: list, agent_id: int):
        self.states = current_states

        meas = []
        time_list = []

        start_time = time.perf_counter()

        while True:
            actions = self.getActions(target_states)

            if actions == (0, 0):
                break

            if self.send_cmd:
                sendCommands([0] * 4 + target_states + [agent_id])
                self.send_cmd = False

            status = self.readTemperature()
            if not status:
                continue

            current_time = time.perf_counter()
            elapsed_time = current_time - start_time
            
            meas.append(self.temp[0])
            time_list.append(elapsed_time)

            self.applyActions(actions)

        self.send_cmd = True

        return [meas, time_list]

    def getActions(self, target_states):
        actions = (self.getAction(self.states[0], target_states[0]),
                   self.getAction(self.states[1], target_states[1]))
        
        return actions

    def getAction(self, state, target_state):
        if state == target_state:
            return 0
        elif state == 0 and target_state == 1:
            return 1
        else:
            return -1
        
    def applyActions(self, actions):
        for i in range(len(actions)):
            self.applyAction(i, actions[i])

    def applyAction(self, i, action):
        if action == 1:
            if self.temp[i] >= self.liquid_threshold:
                self.states[i] = 1
            else:
                print(f'Switching segment {i+1} to soft...')
                print(f'Current temp: {self.temp[i]}')
                print()
        if action == -1:
            if self.temp[i] <= self.solid_threshold:
                self.states[i] = 0
            else:
                print(f'Switching segment {i+1} to rigid...')
                print(f'Current temp: {self.temp[i]}')
                print()


    def readTemperature(self) -> bool:
            response = serial_port.readline()
            response = response.decode('ascii', errors="ignore")

            try:
                temperature = float(response[1:])

                i = -1
                if response[0] == 'A':
                    i = 0
                if response[0] == 'B':
                    i = 1

                if i != -1:
                    self.temp[i] = temperature
                else:
                    return False
            except ValueError:
                return False
            
            return True
