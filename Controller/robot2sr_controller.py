import serial
import numpy as np
from typing import List
from cvxopt import matrix, solvers
from Model import global_var, robot2sr, splines
import sys

port_name = "COM3"

WHEEL_MIN_SPEED = 1
WHEEL_MAX_SPEED = 15

class Controller:
    def __init__(self) -> None:
        self.serial_port = serial.Serial(port_name, 115200)
        self.velocity_coef = 25
        self.prev_v = np.zeros(5)
        self.smoothing_factor = 1
        self.max_linear_velocity = 0.2  # Maximum desired linear velocity
        self.max_angular_velocity = 1.0  # Maximum desired angular velocity
        self.lookahead_distance = 0.05  # Adjust based on your robot's size and desired responsiveness
        self.kp_linear = 3  # Proportional gain for linear velocity
        self.kp_angular = 1  # Proportional gain for angular velocity

    def motionPlanner(self, agent: robot2sr.Robot, path: splines.Trajectory) -> tuple[List[float], List[float]]:
        v = [0] * 5
        s = [0] * 2

        # Find the target point using pure pursuit
        target_point = path.getTargetPoint(agent.position, self.lookahead_distance)

        # Calculate desired heading
        desired_heading = np.arctan2(target_point[1] - agent.y, target_point[0] - agent.x) - np.pi/2
        desired_heading %= (2 * np.pi)

        # Calculate distance to target point
        distance = target_point - np.array(agent.position)
        heading_error = self.min_angle_distance(desired_heading, agent.theta)

        # Calculate linear velocity
        q_tilda_pos = self.kp_linear * distance
        q_tilda_ang = self.kp_angular * heading_error

        q_tilda = q_tilda_pos.tolist() + [q_tilda_ang] + [0, 0]
        v_optimal: np.ndarray = np.matmul(np.linalg.pinv(agent.jacobian), q_tilda)
        
        
        # linear_velocity = min(self.kp_linear * distance, self.max_linear_velocity)

        # Calculate angular velocity
        
        # angular_velocity = self.kp_angular * heading_error
        # if angular_velocity < 0:
        #     angular_velocity = max(angular_velocity, -self.max_angular_velocity)
        # else:
        #     angular_velocity = min(angular_velocity, self.max_angular_velocity)

        # Construct v_optimal
        # v_optimal = np.array([0, 0,
        #                       linear_velocity * np.cos(agent.theta),
        #                       linear_velocity * np.sin(agent.theta),
        #                       0])
        
        v = v_optimal.tolist()

        # Apply constraints to v_optimal
        # v_constrained = self.apply_velocity_constraints(agent, v_optimal)
        
        # v = v_constrained.tolist()

        return v, s


    # def motionPlanner(self, agent: robot2sr.Robot, target: np.ndarray) -> tuple[List[float], List[float]]:
    #     v = [0] * 5
    #     s = [0] * 2

    #     # INVERSE KINEMATICS
    #     dist = target - agent.config
    #     dist_angle = self.min_angle_distance(agent.theta, target[2])
    #     dist[2] = dist_angle

    #     q_tilda = self.velocity_coef * dist * global_var.DT
    #     v_optimal: np.ndarray = np.matmul(np.linalg.pinv(agent.jacobian), q_tilda)

    #     # Apply constraints and smoothing to control velocities
    #     v_constrained = self.apply_velocity_constraints(agent, v_optimal)
    #     # v = v_constrained.tolist()

    #     v_smoothed = self.smooth_velocities(v_constrained)
    #     v = v_smoothed.tolist()

    #     return v, s

    def apply_velocity_constraints(self, agent: robot2sr.Robot, v_optimal: np.ndarray) -> np.ndarray:
        # Define wheel velocity limits
        w_min, w_max = 1.0, 15.0

        # First, constrain control velocities
        v_constrained = np.clip(v_optimal, -self.max_linear_velocity, self.max_linear_velocity)
        v_constrained[2:5] = np.clip(v_constrained[2:5], -self.max_angular_velocity, self.max_angular_velocity)

        # Calculate wheel velocities
        head_wheels, _ = self.__calcWheelsCoords(agent.pose, agent.head.pose)
        tail_wheels, _ = self.__calcWheelsCoords(agent.pose, agent.tail.pose, lu_type='tail')
        wheels = head_wheels + tail_wheels
        V = self.__wheelsConfigMatrix(wheels, agent.stiffness)
        
        wheel_velocities = np.dot(V, v_constrained)

        # Check if any wheel velocity exceeds the limits
        if np.any(wheel_velocities < w_min) or np.any(wheel_velocities > w_max):
            # Scale wheel velocities if limits are exceeded
            scale = min(
                w_max / np.maximum(np.abs(wheel_velocities), 1e-6).max(),
                w_min / np.minimum(np.abs(wheel_velocities), 1e-6).min()
            )
            wheel_velocities *= scale
            
            # Recalculate control velocities from scaled wheel velocities
            v_constrained = np.linalg.lstsq(V, wheel_velocities, rcond=None)[0]

        # print("Original v:", v_optimal)
        # print("Constrained v:", v_constrained)
        # print("Wheel velocities:", wheel_velocities)

        return v_constrained
    
    # def smooth_velocities(self, v_new: np.ndarray) -> np.ndarray:
    #     v_smoothed = self.smoothing_factor * v_new + (1 - self.smoothing_factor) * self.prev_v
    #     self.prev_v = v_smoothed
    #     return v_smoothed
        
    def getWheelsVelocities(self, agent: robot2sr.Robot, v, s) -> list:
        head_wheels, head_wheels_original = self.__calcWheelsCoords(agent.pose, agent.head.pose)
        tail_wheels, tail_wheels_original = self.__calcWheelsCoords(agent.pose, agent.tail.pose, lu_type='tail')
        wheels_global = head_wheels_original + tail_wheels_original

        wheels = head_wheels + tail_wheels

        omega = np.matmul(self.__wheelsConfigMatrix(wheels, s), v).round(3)

        # omega_absolute:np.ndarray = np.abs(omega)

        # gain_coef = WHEEL_MAX_SPEED / omega_absolute.max()
        # if omega_absolute.min() < WHEEL_MIN_SPEED:
        #     if WHEEL_MIN_SPEED / omega_absolute.min() <= gain_coef:
        #         gain_coef = WHEEL_MIN_SPEED / omega_absolute.min()
        #         omega *= gain_coef

        # for i in range(len(omega)):
        #     sign = np.sign(omega[i])
        #     if abs(omega[i]) < 1:
        #         omega[i] = 1

        v_new = np.matmul(np.linalg.pinv(self.__wheelsConfigMatrix(wheels, s)), omega)
        q_dot = np.matmul(agent.jacobian, v_new)
        q = agent.config + q_dot * global_var.DT

        print(omega)

        return omega, wheels_global, q

    def move(self, agent: robot2sr.Robot, v, s) -> list:
        omega, wheels, q = self.getWheelsVelocities(agent, v, s)
        
        commands = omega.tolist() + s + [agent.id]
        # print(commands)

        self.__sendCommands(commands)

        return wheels, q

    def stop(self, agent: robot2sr.Robot) -> None:   
        commands = [0, 0, 0, 0] + agent.stiffness + [agent.id]
        self.__sendCommands(commands)

    def min_angle_distance(self, initial_angle, target_angle):
        # Calculate the clockwise and counterclockwise distances
        clockwise_distance = (target_angle - initial_angle) % (2 * np.pi)
        counterclockwise_distance = (initial_angle - target_angle) % (2 * np.pi)
        
        if clockwise_distance < counterclockwise_distance:
            return -clockwise_distance
        
        # Return the minimum of the two distances
        # return min(clockwise_distance, counterclockwise_distance)
        return counterclockwise_distance

    def __calcWheelsCoords(self, agent_pose: list, lu_pose: list, lu_type='head'):
        if lu_type == 'head':
            w1_0 = 2 * np.array([[-0.0275], [0]])
            w2_0 = 2 * np.array([[0.0105], [-0.0275]])
        elif lu_type == 'tail':
            w1_0 = 2 * np.array([[0.0275], [0]])
            w2_0 = 2 * np.array([[-0.0105], [-0.027]])

        R = np.array([[np.cos(lu_pose[2]), -np.sin(lu_pose[2])],
                    [np.sin(lu_pose[2]), np.cos(lu_pose[2])]])
        w1 = np.matmul(R, w1_0).T[0] + lu_pose[:2]
        w2 = np.matmul(R, w2_0).T[0] + lu_pose[:2]

        w = self.__wheelsToBodyFrame(agent_pose, [w1, w2], lu_pose[-1], lu_type)

        return w, [w1, w2]

    def __wheelsToBodyFrame(self, agent_pose: list, w: list, lu_theta: float, lu_type='head'):
        R_ob = np.array([[np.cos(agent_pose[2]), -np.sin(agent_pose[2])],
                        [np.sin(agent_pose[2]), np.cos(agent_pose[2])]])
        
        T_ob = np.block([[R_ob, np.array([agent_pose[:2]]).T], [np.zeros((1,2)), 1]])
        T_bo = np.linalg.inv(T_ob)

        if lu_type == 'head':
            offset = 0
        elif lu_type == 'tail':
            offset = 2

        for i in range(2):
            w_b0 = [w[i][0], w[i][1], 1]
            w[i] = np.matmul(T_bo, w_b0).T[:-1]
            w[i] = np.append(w[i], (lu_theta - agent_pose[2]) % (2 * np.pi) + global_var.BETA[i+offset])

        return w
    
    def __wheelsConfigMatrix(self, wheels, s) -> np.ndarray:
        flag_soft = int(s[0] or s[1])
        flag_rigid = int(not (s[0] or s[1]))

        V_ = np.zeros((4, 5))
        for i in range(4):
            w = wheels[i]
            tau = w[0] * np.sin(w[-1]) - w[1] * np.cos(w[-1])
            V_[i, :] = [flag_soft * int(i == 0 or i == 1), -flag_soft * int(i == 2 or i == 3), 
                        flag_rigid * np.cos(w[-1]), flag_rigid * np.sin(w[-1]), flag_rigid * tau]

        V = 1 / global_var.WHEEL_R * V_

        return V

    def __sendCommands(self, commands):
        msg = "s"

        for command in commands:
            msg += str(command) + '\n'

        self.serial_port.write(msg.encode())