import robot_controller
import robot_keyboard
import motive_client
from pynput import keyboard
import numpy as np
import math
from nat_net_client import NatNetClient
import sys
from threading import Thread
import queue
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Constants
OMNI_SPEED = 0.12
ROTATION_SPEED = 1
LU_SPEED = 0.12

AGENT_ID = 1

# Length and width of the LU  
a = 0.042 
# Distance between LU center and its corner
r = a * math.sqrt(2) / 2 
# Angle between LU orientation and r
alpha = math.radians(-135)

manual_controller = None


class ManualController(robot_keyboard.ActionsHandler):

    def __init__(self, omni_speed, rotation_speed, lu_speed) -> None:
        super().__init__(omni_speed, rotation_speed, lu_speed)

        self.mocap_data = None
        self.robot_config_queue = queue.Queue()

    def executeAction(self):
        try:
            robot_config = motive_client.getRobotConfig([self.mocap_data])

            if robot_config is not None:
                self.robot_config_queue.put(robot_config)

                omega = robot_controller.getOmega(
                    robot_config[3], self.v, self.s)
                robot_controller.moveRobot(omega, self.s, AGENT_ID)
        except Exception as e:
            print(f"Error occurred: {e}. The robot is stopped!")
            robot_controller.moveRobot(
                np.array([0, 0, 0, 0]), self.s, AGENT_ID)
            
    def plotMotion(self):
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize = (10, 5))  # Initialize a plot

        while True:
            if not self.robot_config_queue.empty(): 
                # Retrieve motion data from the queue
                markers, all_frames, wheels_global, wheels_bf = self.robot_config_queue.get()  

                # Unpack frames
                LU_head_frame, LU_tail_frame, body_frame = all_frames

                # Update the plot with new motion data 
                axs[0].clear()
                axs[1].clear()

                markers_x = [marker.get('marker_x') for marker in markers.values()]
                markers_y = [marker.get('marker_y') for marker in markers.values()]

                axs[0].scatter(markers_x, markers_y)

                # Plot the block of the head LU
                LU_head_rect = (LU_head_frame[0] + r*np.cos(LU_head_frame[2] + alpha), LU_head_frame[1] + r*np.sin(LU_head_frame[2] + alpha))
                axs[0].add_patch(Rectangle(LU_head_rect, a, a, angle=math.degrees(LU_head_frame[2]), edgecolor='black', facecolor='none'))

                # Plot the block of the tail LU
                LU_tail_centre = (LU_tail_frame[0] + r*np.cos(LU_tail_frame[2] + alpha), LU_tail_frame[1] + r*np.sin(LU_tail_frame[2] + alpha))
                axs[0].add_patch(Rectangle(LU_tail_centre, a, a, angle=math.degrees(LU_tail_frame[2]), edgecolor='black', facecolor='none'))

                # Plot all frames
                for frame in all_frames:
                    axs[0].plot(frame[0], frame[1], 'r*')
                    axs[0].plot([frame[0], frame[0] + 0.03 * np.cos(frame[2])], [frame[1], frame[1] + 0.03 * np.sin(frame[2])], 'r')

                # Plot wheels
                for wheel in wheels_global:
                    axs[0].plot(wheel[0], wheel[1], 'mo', markersize=15)

                # Plot the bridge curve
                vsf_markers = [marker for marker in markers.values() if marker['model_id'] == 0 and marker['rank'] != 6]
                vsf_markers.sort(key=lambda marker: marker['rank'])

                vsf_markers_x = [vsf_marker['marker_x'] for vsf_marker in vsf_markers]
                vsf_markers_y = [vsf_marker['marker_y'] for vsf_marker in vsf_markers]

                axs[0].plot(vsf_markers_x, vsf_markers_y, color='orange')

                # Connect the bridge with the LU's
                vsf_start = (LU_head_frame[0] + r*np.cos(LU_head_frame[2] + alpha - np.pi), LU_head_frame[1] + r*np.sin(LU_head_frame[2] + alpha - np.pi))
                axs[0].plot([vsf_start[0], vsf_markers_x[0]], [vsf_start[1], vsf_markers_y[0]], color='orange')

                vsf_end = (LU_tail_frame[0] + r*np.cos(LU_tail_frame[2] + alpha - np.pi/2), LU_tail_frame[1] + r*np.sin(LU_tail_frame[2] + alpha - np.pi/2))
                axs[0].plot([vsf_end[0], vsf_markers_x[-1]], [vsf_end[1], vsf_markers_y[-1]], color='orange')

                axs[0].axis('equal')


                # PRINT THE ROBOT IN A BODY FRAME

                axs[1].plot(0, 0, 'r*')
                axs[1].plot([0, 0.05 * np.cos(0)], [0, 0.05 * np.sin(0)], 'r')

                for wheel in wheels_bf:
                    axs[1].plot(wheel[0], wheel[1], 'mo', markersize=15)
                    axs[1].plot([wheel[0], wheel[0] + 0.03 * np.cos(wheel[2])], [wheel[1], wheel[1] + 0.03 * np.sin(wheel[2])], 'r')
                
                axs[1].axis('equal')

                plt.pause(0.1)  

    def onPress(self, key) -> None:
        super().onPress(key)
        self.executeAction()

    def onRelease(self, key) -> None:
        super().onRelease(key)
        robot_controller.moveRobot(np.array([0, 0, 0, 0]), self.s, AGENT_ID)


def receiveMocapDataFrame(data):
    manual_controller.mocap_data = data


def parseArgs(arg_list, args_dict):
    arg_list_len = len(arg_list)
    if arg_list_len > 1:
        args_dict["serverAddress"] = arg_list[1]
        if arg_list_len > 2:
            args_dict["clientAddress"] = arg_list[2]
        if arg_list_len > 3:
            if len(arg_list[3]):
                args_dict["use_multicast"] = True
                if arg_list[3][0].upper() == "U":
                    args_dict["use_multicast"] = False
    return args_dict


if __name__ == "__main__":
    manual_controller = ManualController(OMNI_SPEED, ROTATION_SPEED, LU_SPEED)

    # Start a thread for real-time plotting
    plot_thread = Thread(target=manual_controller.plotMotion)
    # Make the plot thread a daemon so it doesn't block program exit
    plot_thread.daemon = True  
    plot_thread.start()

    options_dict = {}
    options_dict["clientAddress"] = "127.0.0.1"
    options_dict["serverAddress"] = "127.0.0.1"
    options_dict["use_multicast"] = True

    # This will create a new NatNet client
    options_dict = parseArgs(sys.argv, options_dict)

    streaming_client = NatNetClient()
    streaming_client.set_client_address(options_dict["clientAddress"])
    streaming_client.set_server_address(options_dict["serverAddress"])
    streaming_client.set_use_multicast(options_dict["use_multicast"])

    streaming_client.mocap_data_listener = receiveMocapDataFrame

    is_running = streaming_client.run()

    with keyboard.Listener(on_press=manual_controller.onPress, on_release=manual_controller.onRelease) as listener:
        listener.join()
