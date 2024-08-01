import serial
import math
import numpy as np
from typing import List
from cvxopt import matrix, solvers
from Model import global_var, robot2sr, splines
import cvxpy

port_name = "COM3"


class Controller:
    def __init__(self) -> None:
        self.serial_port = serial.Serial(port_name, 115200)

        self.max_linear_velocity = 0.2  # Maximum desired linear velocity
        self.max_angular_velocity = 1.0  # Maximum desired angular velocity
        self.lookahead_distance = 0.05  # Adjust based on your robot's size and desired responsiveness
        self.kp_linear = 3  # Proportional gain for linear velocity
        self.kp_angular = 1  # Proportional gain for angular velocity

        self.dt = global_var.DT

        self.sp = None

        self.NX = 3  # x = x, y, yaw
        self.NU = 2  # a = [linear velocity, linear velocity, angular velocity ]
        self.NW = 4  # number of wheels
        self.T = 10  # horizon length

        # mpc parameters
        self.R = np.diag([10000, .05])  # input cost matrix
        self.Q = np.diag([10, 10, 0.0])  # agent cost matrix
        self.Qf = self.Q # agent final matrix
        self.Rd = np.diag([10000, 0.01])

        self.TARGET_SPEED = 0.08
        self.MAX_SPEED = 15  # Maximum speed in rad/s
        self.MIN_SPEED = 0.5  # Minimum non-zero speed in rad/s 

    def motionPlannerMPC(self, agent: robot2sr.Robot, path: splines.Trajectory, v_current) -> tuple[List[float], List[float]]:        
        cx, cy, cyaw, s = path.params

        if self.sp is None:
            self.sp = []
            for i in range(1, len(cx)):
                self.sp.append(self.TARGET_SPEED * (1 - path.curvature[i]/(max(path.curvature)+5)))
            self.sp.append(0)

        target_ind = path.getTarget(agent.position, self.lookahead_distance)
        qref, vref = self.__calc_ref_trajectory(cx, cy, cyaw, self.sp, target_ind, v_current[0]) 
        qbar = self.__predict_motion(agent.config, qref.shape, [v_current[0]] * self.T, [v_current[1]] * self.T)

        head_wheels, _ = self._calcWheelsCoords(agent.pose, agent.head.pose)
        tail_wheels, _ = self._calcWheelsCoords(agent.pose, agent.tail.pose, lu_type='tail')
        wheels = head_wheels + tail_wheels

        v_predicted, omega_predicted = self.__linear_mpc_control(qref, qbar, agent.pose, vref, wheels)

        return [0, v_predicted[0], omega_predicted[0], 0, 0], agent.stiffness
    
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
    
    def __predict_motion(self, agent_config: list, shape: tuple, v_list: list, omega_list: list) -> np.ndarray:
        qbar = np.zeros(shape)
        qbar[:, 0] = agent_config[:3]

        agent = robot2sr.Robot(1, *agent_config)
        self.__update_agent(agent, agent.config)
        for (i, v, omega) in zip(range(1, self.T + 1), v_list, omega_list):
            q = self.get_config(agent, [0, v, omega, 0, 0], agent.stiffness)
            self.__update_agent(agent, q)
            qbar[:, i] = agent.pose

        return qbar
    
    def __update_agent(self, agent: robot2sr.Robot, q: np.ndarray) -> None:
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
    
    def __linear_mpc_control(self, qref: np.ndarray, qbar: np.ndarray, agent_pose: list, vref: np.ndarray, wheels: list):
        q = cvxpy.Variable((self.NX, self.T + 1))
        u = cvxpy.Variable((self.NU, self.T))
        vw = cvxpy.Variable((self.NW, self.T)) #to calculate vr and vl

        # Add binary variables for each wheel at each time step
        z = cvxpy.Variable((self.NW, self.T), boolean=True)
    
        cost = 0.0
        constraints = []

        for t in range(self.T):
            cost += cvxpy.quad_form(u[:, t], self.R)
            if t != 0:
                cost += cvxpy.quad_form(q[:, t], self.Q)        
            A, B = self.__get_linear_model_matrix(vref[0, t], qbar[2, t])  

            constraints += [q[:, t + 1] == A @ q[:, t] + B @ u[:, t]]  

            for w in range(self.NW):
                constraints += [vw[w, t] == (1 / global_var.WHEEL_R) * (math.sin(wheels[w][2]) * (u[0, t] + vref[0, t+1]) + 
                                                            (wheels[w][0] * math.sin(wheels[w][2]) - wheels[w][1] * math.cos(wheels[w][2])) * u[1, t])]

            if t < (self.T - 1):
                cost += cvxpy.quad_form((u[:, t + 1] - u[:, t]), self.Rd)

        # constraints += [vw[:,:] <= 15]  
        # constraints += [vw[:,:] >= -15]    

        for w in range(self.NW):
            # Exclude velocities from -1 to 1
            constraints += [vw[w, :] >= self.MIN_SPEED - (self.MAX_SPEED + self.MIN_SPEED) * z[w, :]]
            constraints += [vw[w, :] <= -self.MIN_SPEED + (self.MAX_SPEED + self.MIN_SPEED) * (1 - z[w, :])]

        cost += cvxpy.quad_form(q[:, self.T], self.Qf)
        constraints += [q[:, 0] ==  agent_pose - qref[:,0]]    
        prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
        prob.solve(solver=cvxpy.ECOS_BB, verbose=False)
        #OSQP,CVXOPT, ECOS, scs


        if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
            v_predicted = np.array(u.value[0, :]).flatten() + vref[0, 1:]
            omega_predicted = np.array(u.value[1, :]).flatten()
        else:
            print("Error: Cannot solve mpc..")
            v_predicted, omega_predicted = None, None

        return v_predicted, omega_predicted
    
    def __get_linear_model_matrix(self, vref, phi):
        A = np.zeros((self.NX, self.NX))
        A[0, 0] = 1.0
        A[0, 2] = -vref * math.cos(phi) * self.dt
        A[1, 1] = 1.0
        A[1, 2] = -vref * math.sin(phi) * self.dt
        A[2, 2] = 1.0

        B = np.zeros((self.NX, self.NU))
        B[0, 0] = -self.dt * math.sin(phi)
        B[0, 1] = 0 #0
        B[1, 0] = self.dt * math.cos(phi)
        B[1, 1] = 0 #0
        B[2, 1] = self.dt

        return A, B


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


    def applyVelocityConstraints(self, agent: robot2sr.Robot, v_optimal: np.ndarray) -> np.ndarray:
        # First, constrain control velocities
        v_constrained = np.clip(v_optimal, -self.max_linear_velocity, self.max_linear_velocity)
        v_constrained[2:5] = np.clip(v_constrained[2:5], -self.max_angular_velocity, self.max_angular_velocity)

        # Calculate wheel velocities
        head_wheels, _ = self._calcWheelsCoords(agent.pose, agent.head.pose)
        tail_wheels, _ = self._calcWheelsCoords(agent.pose, agent.tail.pose, lu_type='tail')
        wheels = head_wheels + tail_wheels
        V = self._wheelsConfigMatrix(wheels, agent.stiffness)
        
        wheel_velocities = np.dot(V, v_constrained)

        # Check if any wheel velocity exceeds the limits
        if np.any(wheel_velocities < WHEEL_MIN_SPEED) or np.any(wheel_velocities > WHEEL_MAX_SPEED):
            # Scale wheel velocities if limits are exceeded
            scale = min(
                WHEEL_MAX_SPEED / np.maximum(np.abs(wheel_velocities), 1e-6).max(),
                WHEEL_MIN_SPEED / np.minimum(np.abs(wheel_velocities), 1e-6).min()
            )
            wheel_velocities *= scale
            
            # Recalculate control velocities from scaled wheel velocities
            v_constrained = np.linalg.lstsq(V, wheel_velocities, rcond=None)[0]

        return v_constrained
    
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

        print(omega)
        return omega, wheels, self.get_config(agent, v_new, s)

    def move(self, agent: robot2sr.Robot, v: List[float], s: List[float]) -> tuple[List[List[float]], List[float]]:
        omega, wheels, q = self.getWheelsVelocities(agent, v, s)
        commands = omega.tolist() + s + [agent.id]
        self._sendCommands(commands)
        return wheels, q

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
                       flag_soft * (i < 2), -flag_soft * (i >= 2)]

        return 1 / global_var.WHEEL_R * V

    def _sendCommands(self, commands: List[float]) -> None:
        msg = "s" + "".join(f"{command}\n" for command in commands)
        self.serial_port.write(msg.encode())