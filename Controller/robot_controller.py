import serial
import numpy as np
import Model.global_var as global_var


port_name = "COM5"
# port_name = "/dev/tty.usbserial-0001"
serial_port = serial.Serial(port_name, 115200)


def getOmega(w_list, v, s):

    # w_list contains coordinates (position and orientation) of wheels

    flag_soft = int(s[0] or s[1])
    flag_rigid = int(not (s[0] or s[1]))

    V_ = np.zeros((4, 5))
    for i in range(4):
        w = w_list[i]
        tau = w[0] * np.sin(w[2]) - w[1] * np.cos(w[2])
        V_[i, :] = [flag_soft * int(i == 0 or i == 1), -flag_soft * int(
            i == 2 or i == 3), flag_rigid * np.cos(w[2]), flag_rigid * np.sin(w[2]), flag_rigid * tau]

    V = 1 / global_var.WHEEL_R * V_
    omega = np.matmul(V, v)

    return omega.round(3)


def moveRobot(omega, s, agent_id):

    commands = omega.tolist() + s + [agent_id]
    print(commands)

    sendData(commands)


def sendData(commands):

    msg = "s"

    for command in commands:
        msg += str(command) + '\n'

    # print(msg.encode())

    serial_port.write(msg.encode())
