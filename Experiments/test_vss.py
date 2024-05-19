import serial
from pynput import keyboard
from threading import Thread
import time 
import pandas as pd
import time
import cv2

portName = "/dev/tty.usbserial-0001"
serial_port = serial.Serial(portName, 115200)

start_time = time.time()
vss_stiffness = 0

T_prev = None

step = 0

heating = True
flexible = False
cooling = False

step_start_time = 0
heating_time = 0
cooling_time = 0

mode = 'heating'

rgb_video_stream_widget = None
thermal_video_stream_widget = None

path = '/Users/lytaura/Documents/PolyU/Research/2SR/Version\ 1/Multi\ agent/Control/2sr-swarm-control/Data/Video/'

def on_press(key):
    global step, start_time, rgb_video_stream_widget, thermal_video_stream_widget

    if key.char == 's':
        print("Start!")
        # vss_stiffness = 1
        step = 1
        start_time = time.time()

    # if key.char == 'x':
    #     print("rigid")
    #     vss_stiffness = 0

    # commands = [0] * 2 + [vss_stiffness]

    # send_commands(commands)

def send_commands(commands):
    msg = "s"

    for command in commands:
        msg += str(command) + '\n'

    # print(msg.encode())

    serial_port.write(msg.encode())


def read_sensor():
    global vss_stiffness, T_prev, step, heating, flexible, cooling, step_start_time, heating_time, cooling_time, mode

    while True:

        if step == 0:
            continue
        if step > 3:
            print('Finish')
            print('')
            print('Heating time: ' + str(heating_time / 3))
            print('Cooling time: ' + str(cooling_time / 3))

            rgb_video_stream_widget.capture.release()
            rgb_video_stream_widget.out.release()

            thermal_video_stream_widget.capture.release()
            thermal_video_stream_widget.out.release()

            break

        # rgb_video_stream_widget.out.write(rgb_video_stream_widget.frame)
        # thermal_video_stream_widget.out.write(thermal_video_stream_widget.frame)

        if heating:
            vss_stiffness = 1
            mode = 'heating'
            print('Heating')
        if flexible:
            vss_stiffness = 1
            mode = 'flexible'
            print('Flexible state')
        if cooling:
            vss_stiffness = 0
            mode = 'cooling'
            print('Cooling')

        send_commands([0, 0, vss_stiffness])

        response = serial_port.readline()
        response = response.decode('ascii', errors="ignore")
        
        try:
            temperature = float(response)
            print(temperature)

            if temperature < 20 or temperature > 150:
                continue

            if T_prev is None:
                T_prev = temperature
            else:
                if abs(T_prev - temperature) > 5:
                    continue

            current_time = time.time() - start_time

            if heating and temperature >= 53.5:
                heating = False
                flexible = True

                heating_time += current_time - step_start_time
                step_start_time = current_time

            if flexible and current_time - step_start_time > 60:
                flexible = False
                cooling = True

                step_start_time = current_time

            if cooling and temperature < 38:
                cooling = False
                heating = True

                cooling_time += current_time - step_start_time
                step_start_time = current_time

                step += 1

            data = {'stiffness': [vss_stiffness], 'T': [temperature], 
                    'mode': [mode], 'time': [current_time]}
            
            df = pd.DataFrame(data)
            df.to_csv('vss_log.csv', mode='a', index=False, header=False)

            T_prev = temperature

        except ValueError:
            pass

class VideoStreamWidget(object):
    def __init__(self, src=0):
        if src == 0:
            self.title = 'rgb'
            self.capture = cv2.VideoCapture(src)
        elif src == 1:
            self.title = 'thermal'
            stream_url = 'http://10.11.44.155:1234/'
            self.capture = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)

        # self.capture.set(3,1080)
        # self.capture.set(4,720)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        w = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.out = cv2.VideoWriter(self.title + '.mp4', fourcc, 20, (w,h))

        # Start the thread to read frames from the video stream
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        # Read the next frame from the stream in a different thread
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
                # cv2.imshow(self.title, self.frame)
                if self.status and step > 0:
                    self.out.write(self.frame)
            # time.sleep(.01)
    
    def show_frame(self):
        # Display frames in main program
        cv2.imshow(self.title, self.frame)
                
        key = cv2.waitKey(1)
        if key == ord('q'):
            self.capture.release()
            self.out.release()
            cv2.destroyWindow(self.title)
            # exit(1)


if __name__ == "__main__":
    f = open('vss_log.csv', "w+")
    f.close()

    rgb_video_stream_widget = VideoStreamWidget()
    thermal_video_stream_widget = VideoStreamWidget(src=1)

    thread = Thread(target = read_sensor)
    thread.start()

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

    # while True:
    #     try:
    #         rgb_video_stream_widget.show_frame()
    #         thermal_video_stream_widget.show_frame()
    #     except AttributeError:
    #         pass

