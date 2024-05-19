import serial
from pynput import keyboard
import numpy as np
import pandas as pd
from collections import deque 
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter

portName = "/dev/tty.usbserial-0001"
serial_port = serial.Serial(portName, 115200)

target = 12.0

data_buff = deque([])

start_time = time.time()

fig, ax = plt.subplots()

line_target = ax.plot([], [], lw=2, label='Target')[0]
line_voltage = ax.plot([], [], lw=2, label='Voltage')[0]
line_velocity = ax.plot([], [], lw=2, label='Velocity')[0]

ax.legend()

def on_press(key):

    w = [0.0, 0.0]

    if key == keyboard.Key.up:
        print("forward")
        w = [target, 0.0]

    if key == keyboard.Key.down:
        print("backward")
        w = [-target, 0.0]

    if key == keyboard.Key.left:
        print("left")
        w = [0.0, target]

    if key == keyboard.Key.right:
        print("right")
        w = [0.0, -target]

    move(w)


def on_release(key): 
    try:
        w = [0.0, 0.0]
        move(w)
    except KeyError:
        pass

def move(commands):
        # print(commands)
        msg = "s"

        for command in commands:
            msg += str(command) + '\n'

        # print(msg.encode())

        serial_port.write(msg.encode())

        response = ''
        motor_data = []

        while not response:
            response = serial_port.readline()
            # print(response)
            response = response.decode('ascii', errors="ignore")
            print(response)
            response_values = response.split()

            if len(response_values) != 4:
                break

            for value in response_values:
                try:
                    motor_value = float(value)
                    motor_data.append(motor_value)
                except ValueError:
                    break

            if len(motor_data) == 4:
                if motor_data[0] != target:
                    continue

                current_time = time.time() - start_time

                data = {'target': [motor_data[0]], 'voltage': [motor_data[1]], 
                        'velocity': [motor_data[2]], 'ange': [motor_data[3]],
                        'time': [current_time]}
                
                # print(data)
                df = pd.DataFrame(data)

                df.to_csv('motor_log.csv', mode='a', index=False, header=False)

                data_buff.append(motor_data)

def run(frame):
    target_velocity = 12.0
    w = [0.0, target_velocity]
    move(w)

    if len(data_buff) > 50:
        data_buff.popleft()

    all_data = np.array(data_buff)
    if len(all_data.shape) == 2:
        xdata = list(range(len(data_buff)))
        
        ydata_target = list(all_data[:,0])
        ydata_voltage = list(all_data[:,1])
        ydata_velocity = list(all_data[:,2])
    else:
        xdata = [0]

        ydata_target = [0]
        ydata_voltage = [0]
        ydata_velocity = [0]

    line_target.set_data(xdata, ydata_target)
    line_voltage.set_data(xdata, ydata_voltage)
    line_velocity.set_data(xdata, ydata_velocity)

    xmin = min(xdata) - 1
    xmax = max(xdata) + 1

    ymin = min(ydata_target + ydata_voltage + ydata_velocity) - 10
    ymax = max(ydata_target + ydata_voltage + ydata_velocity) + 10

    ylim = ax.get_ylim()
    if ylim[0] - ymin > 100:
        ymin = ylim[0]
    if ymax - ylim[1] > 100:
        ymax = ylim[1]

    ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax])

    return (line_target, line_voltage, line_velocity)


if __name__ == "__main__":
    # with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    #     listener.join()

    f = open('Data/motor_log.csv', "w+")
    f.close()

    # anim = animation.FuncAnimation(fig, run, frames=1000, interval=10)

    # mywriter = FFMpegWriter(fps=30)
    # anim.save('Data/Video/motor_vis2.mp4', writer=mywriter, dpi=300)

    # plt.show()

    while True:
        target_velocity = 12.0
        w = [0.0, target_velocity]
        move(w)
