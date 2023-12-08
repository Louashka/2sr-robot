import serial
import numpy as np
from typing import List
from Model import global_var, agent
from typing import List


port_name = "COM5"
# port_name = "/dev/tty.usbserial-0001"
serial_port = serial.Serial(port_name, 115200)

class Swarm:
    def __init__(self) -> None:
        self.__agents = []

    @property
    def agents(self) -> List[agent.Robot]:
        return self.__agents
    
    @agents.setter
    def agents(self, value: List[agent.Robot]) -> None:
        if not isinstance(value, List[agent.Robot]):
            raise Exception('Wrong type of agent!')
        self.__agents = value

    def getAllId(self) -> List[int]:
        all_id = []
        if self.agents is not None:
            for agent in self.agents:
                all_id.append(agent.id)

        return all_id
    
    def getAgentById(self, id) -> agent.Robot:
        for agent in self.agents:
            if agent.id == id:
                return agent
            
        return None
    
    def getActiveAgents(self) -> List[agent.Robot]:
        active_agents = []
        for agent in self.agents:
            if agent.status:
                active_agents.append(agent)

        return active_agents
    
    def move(self, v, s) -> None:
        for agent in self.agents:
            omega = self.__getAgentOmega(agent.allWheels, v, s)
            self.__moveAgent(agent.id, omega, s)

    def stop(self) -> None:
        for agent in self.agents:
            self.__moveAgent(agent.id, np.array([0, 0, 0, 0]), [0, 0])

    def __getAgentOmega(self, wheels: List[agent.Wheel], v, s):
        # w_list contains coordinates (position and orientation) of wheels

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



