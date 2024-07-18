import serial
import numpy as np
from typing import List
from cvxopt import matrix, solvers
from Model import global_var, robot2sr, splines

port_name = "COM3"

# Define wheel velocity limits
WHEEL_MIN_SPEED = 1
WHEEL_MAX_SPEED = 15

class Controller:
    def __init__(self) -> None:
        self.serial_port = serial.Serial(port_name, 115200)

        self.max_linear_velocity = 0.2  # Maximum desired linear velocity
        self.max_angular_velocity = 1.0  # Maximum desired angular velocity
        self.lookahead_distance = 0.05  # Adjust based on your robot's size and desired responsiveness
        self.kp_linear = 3  # Proportional gain for linear velocity
        self.kp_angular = 1  # Proportional gain for angular velocity

    def motionPlanner(self, agent: robot2sr.Robot, path: splines.Trajectory) -> tuple[List[float], List[float]]:
        target_point = path.getTargetPoint(agent.position, self.lookahead_distance)
        desired_heading = np.arctan2(target_point[1] - agent.y, target_point[0] - agent.x) - np.pi/2
        desired_heading %= (2 * np.pi)

        distance = target_point - np.array(agent.position)
        heading_error = self.minAngleDistance(desired_heading, agent.theta)

        q_tilda = np.concatenate([
            self.kp_linear * distance,
            [self.kp_angular * heading_error],
            [0, 0]
        ])

        v_optimal = np.linalg.pinv(agent.jacobian) @ q_tilda

        # Apply constraints to v_optimal
        # v_constrained = self.applyCelocity_constraints(agent, v_optimal)
        
        # v = v_constrained.tolist()
        
        return v_optimal.tolist(), agent.stiffness


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
    
    def getWheelsVelocities(self, agent: robot2sr.Robot, v: List[float], s: List[float]) -> tuple[np.ndarray, List[List[float]], np.ndarray]:
        head_wheels, head_wheels_global = self._calcWheelsCoords(agent.pose, agent.head.pose)
        tail_wheels, tail_wheels_global = self._calcWheelsCoords(agent.pose, agent.tail.pose, lu_type='tail')
        wheels = head_wheels + tail_wheels
        wheels_global = head_wheels_global + tail_wheels_global

        V = self._wheelsConfigMatrix(wheels, s)
        omega = np.round(V @ v, 3)
        v_new = np.linalg.pinv(V) @ omega
        q_dot = agent.jacobian @ v_new
        q = agent.config + q_dot * global_var.DT

        print(omega)
        return omega, wheels_global, q

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
        w1_0, w2_0 = 2 * w1_0, 2 * w2_0

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
        flag_soft = int(any(s))
        flag_rigid = int(not any(s))

        V = np.zeros((4, 5))
        for i, w in enumerate(wheels):
            tau = w[0] * np.sin(w[-1]) - w[1] * np.cos(w[-1])
            V[i, :] = [flag_rigid * np.cos(w[-1]), flag_rigid * np.sin(w[-1]), flag_rigid * tau,
                       flag_soft * (i < 2), -flag_soft * (i >= 2)]

        return 1 / global_var.WHEEL_R * V

    def _sendCommands(self, commands: List[float]) -> None:
        msg = "s" + "".join(f"{command}\n" for command in commands)
        self.serial_port.write(msg.encode())