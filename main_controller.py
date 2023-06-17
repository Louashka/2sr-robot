import serial
import numpy as np
import globals_


class Controller:

    def __init__(self, port_name):
        self.serial_port = serial.Serial(port_name, 115200)

    def wheel_drive(self, w_list, v, s):

        flag_soft = int(s[0] or s[1])
        flag_rigid = int(not (s[0] or s[1]))

        V_ = np.zeros((4, 5))
        for i in range(4):
            w = w_list[i]
            tau = w[0] * np.sin(w[2]) - w[1] * np.cos(w[2])
            V_[i, :] = [flag_soft * int(i == 0), -flag_soft * int(
                i == 2), flag_rigid * np.cos(w[2]), flag_rigid * np.sin(w[2]), flag_rigid * tau]

        V = 1 / globals_.WHEEL_R * V_
        omega = np.matmul(V, v)

        return omega.round(3)

    def move_robot(self, omega, s, agent_id):

        commands = omega.tolist() + s + [agent_id]
        print(commands)

        self.send_data(commands)

    def send_data(self, commands):

        msg = "s"

        for command in commands:
            msg += str(command) + '\n'

        # print(msg.encode())

        self.serial_port.write(msg.encode())
