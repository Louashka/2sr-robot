import serial
import numpy as np
from typing import List
from Model import global_var, robot2sr
from typing import List


# port_name = "COM5"
port_name = "/dev/tty.usbserial-0001"
serial_port = serial.Serial(port_name, 115200)

class Swarm:
    def __init__(self) -> None:
        self.__agents = []

    @property
    def agents(self) -> List[robot2sr.Robot]:
        return self.__agents
    
    @agents.setter
    def agents(self, value: List[robot2sr.Robot]) -> None:
        if not isinstance(value, List[robot2sr.Robot]):
            raise Exception('Wrong type of agent!')
        self.__agents = value

    def getAllId(self) -> List[int]:
        all_id = []
        if self.agents is not None:
            for agent in self.agents:
                all_id.append(agent.id)

        return all_id
    
    def getAgentById(self, id) -> robot2sr.Robot:
        for agent in self.agents:
            if agent.id == id:
                return agent
            
        return None
    
    def getActiveAgents(self) -> List[robot2sr.Robot]:
        active_agents = []
        for agent in self.agents:
            if agent.status:
                active_agents.append(agent)

        return active_agents
    
    def move(self, v, s) -> None:
        for agent in self.agents:
            # omega = self.__getAgentOmega(agent.allWheels, v, s)
            # self.__moveAgent(agent.id, omega, s)
            pass

    def stop(self) -> None:
        for agent in self.agents:
            self.__moveAgent(agent.id, np.array([0, 0, 0, 0]), [0, 0])    


    def __getAgentOmega(self, wheels, v, s):

        flag_soft = int(s[0] or s[1])
        flag_rigid = int(not (s[0] or s[1]))

        V_ = np.zeros((4, 5))
        for i in range(4):
            w = wheels[i]
            tau = w.x * np.sin(w.theta) - w.y * np.cos(w.theta)
            V_[i, :] = [flag_soft * int(i == 0 or i == 1), -flag_soft * int(
                i == 2 or i == 3), flag_rigid * np.cos(w.theta), flag_rigid * np.sin(w.theta), flag_rigid * tau]

        V = 1 / global_var.WHEEL_R * V_
        omega = np.matmul(V, v)

        return omega.round(3)

    def __moveAgent(self, agent_id, omega, s):
        commands = omega.tolist() + s + [agent_id]
        print(commands)

        self.__sendCommands(commands)

    def __sendCommands(self, commands):
        msg = "s"

        for command in commands:
            msg += str(command) + '\n'

        serial_port.write(msg.encode())

    def ik(self, robot_id: int, s_current: list, q_d: list, total_error: float) -> tuple[list, list, float]:
        robot = self.getAgentById(robot_id)
        q_d = np.array(q_d)
        error = np.linalg.norm(q_d - robot.config)

        coef = 0.5  # Feedback gain
        dt = 0.5  # Step size
        # A set of possible stiffness configurations
        s_options = [[0, 0], [0, 1], [1, 0], [1, 1]]
        i_current = s_options.index(s_current)

        # A set of possible configurations and velocities
        q_ = [None] * len(s_options)
        v_ = [None] * len(s_options)

        q_tilda = (q_d - robot.config) * coef
        for i in range(len(s_options)):
            # Jacobian matrix
            Jacobian = self.__hybridJacobian(robot.config, s_options[i])
            # velocity input commands
            v_[i] = np.matmul(np.linalg.pinv(Jacobian), q_tilda)
            q_dot_desired = np.matmul(Jacobian, v_[i])
            q_dot = self.__piFeedback(error, total_error, q_dot_desired)
            q_[i] = robot.config + q_dot * dt

        # Determine the stiffness configuration that promotes faster approach to the target
        error_ = np.linalg.norm(q_ - q_d, axis=1)
        min_i = np.argmin(error_)

        # The extent of the configuration change
        delta_q_ = np.linalg.norm(robot.config - np.array(q_), axis=1)

        # Stiffness transition is committed only if the previous configuration does not promote further motion
        if min_i != i_current and delta_q_[i_current].all() > 10**(-2):
            min_i = i_current

        v = v_[min_i]
        s = s_options[min_i]

        return v, s, error

    def __hybridJacobian(self, q: list, s: list) -> np.ndarray:
        # We divide a Jacobian matrix into 2 parts, which correspond
        # to the rigid and soft states of the 2SR robot

        # RIGID STATE

        flag_rigid = int(not (s[0] or s[1]))  # checks if both
        # segments are rigid
        J_rigid = flag_rigid * np.array([[np.cos(q[2]), -np.sin(q[2]), 0],
                                        [np.sin(q[2]), np.cos(q[2]), 0],
                                        [0, 0, 1],
                                        [0, 0, 0],
                                        [0, 0, 0]])

        # SOFT STATE

        # Central angles of VSS arcs
        alpha = np.array(q[3:]) * global_var.L_VSS
        # Angles of 3 types of logarithmic spirals
        theta = np.divide(np.tile(np.reshape(alpha, (2, 1)), 3), global_var.M) - np.pi

        # Spiral constants
        a = global_var.SPIRAL_COEF[0]
        b = np.tile(global_var.SPIRAL_COEF[1], (2, 1))
        for i in range(2):
            if q[i + 3] > 0:
                b[i] = -b[i]
                theta[i] += 2 * np.pi

        rho = a * np.exp(b * theta)  # spiral radii

        # Proportionality coefficient of the rate of VSS curvature change
        Kappa = np.divide(global_var.M, global_var.L_VSS * rho)
        # Proportionality coefficient of the rate of 2SRR orientation change
        Phi = np.divide(global_var.M, rho)

        flag_soft1 = int(not s[1] and s[0])  # checks if VSS1 is soft
        flag_soft2 = int(not s[0] and s[1])  # checks if VSS2 is soft
        flag_full_soft = int(s[0] and s[1])  # checks if both VSSs are soft

        #  Proportionality coefficients of the rate of change of 2SRR position coordinates
        Delta = np.zeros((2, 2))
        Delta[:, 0] = (flag_soft2 * Kappa[1, 1] + flag_full_soft * Kappa[1, 2]) * \
            self.__pDerivPos(rho[1, 0], q[2], q[4], q[4], 2).ravel()
        Delta[:, 1] = (flag_soft1 * Kappa[0, 1] + flag_full_soft * Kappa[0, 2]) * \
            self.__pDerivPos(rho[0, 0], q[2], q[3], q[3], 1).ravel()

        J_soft = np.array([[-flag_soft2 * Phi[1, 1] - flag_full_soft * Phi[1, 2],
                            flag_soft1 * Phi[0, 1] + flag_full_soft * Phi[0, 2]],
                        [flag_soft1 * Kappa[0, 0] + flag_full_soft * Kappa[0, 2],
                            flag_soft1 * Kappa[0, 1] + flag_full_soft * Kappa[0, 2]],
                        [flag_soft2 * Kappa[1, 1] + flag_full_soft * Kappa[1, 2],
                            flag_soft2 * Kappa[1, 0] + flag_full_soft * Kappa[1, 2]]])

        J_soft = np.concatenate((Delta, J_soft), axis=0)

        # complete hybrid Jacobian matrix
        J = np.concatenate((J_soft, J_rigid), axis=1)

        return J
    
    def __pDerivPos(self, r: float, phi: float, k: float, k0: float, seg: int) -> np.ndarray:
        if seg == 1:
            pos = np.array([[-0.0266 * r * np.sin(phi + 0.0266 * k - 0.04 * k0) - 0.0006 * np.sin(phi + 0.04 * k - 0.04 * k0)],
                            [0.0266 * r * np.cos(phi + 0.0266 * k - 0.04 * k0) + 0.0006 * np.cos(phi + 0.04 * k - 0.04 * k0)]])
            # pos = bodyFramePosition(-1)
        elif seg == 2:
            pos = np.array([[-0.0266 * r * np.sin(phi - 0.0266 * k + 0.04 * k0) - 0.0006 * np.sin(phi - 0.04 * k + 0.04 * k0)],
                            [0.0266 * r * np.cos(phi - 0.0266 * k + 0.04 * k0) + 0.0006 * np.cos(phi - 0.04 * k + 0.04 * k0)]])
            
        return pos

    def __piFeedback(self, error: float, total_error: float, q_dot_desired: np.ndarray) -> np.ndarray:
        kp = 0.5  # Proportional gain
        ki = 0.1  # Integral gain

        feedback = kp * np.eye(5, dtype=int) * error + ki * np.eye(5, dtype=int) * total_error
        q_dot = q_dot_desired + feedback

        return q_dot
    
