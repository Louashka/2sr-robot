import robot_controller
import robot_keyboard
import motive_client
from pynput import keyboard
import numpy as np
from nat_net_client import NatNetClient
import sys

# Constants
OMNI_SPEED = 0.12
ROTATION_SPEED = 1
LU_SPEED = 0.12

AGENT_ID = 1


class ManualController(robot_keyboard.ActionsHandler):

    def __init__(self, omni_speed, rotation_speed, lu_speed) -> None:
        super().__init__(omni_speed, rotation_speed, lu_speed)

        self.mocap_data = None

    def executeAction(self):
        try:
            w_current = motive_client.getWheelsCoords([self.mocap_data])

            if w_current is not None:
                omega = robot_controller.getOmega(
                    w_current, self.v, self.s)
                robot_controller.moveRobot(omega, self.s, AGENT_ID)
        except Exception as e:
            print(f"Error occurred: {e}. The robot is stopped")
            robot_controller.moveRobot(
                np.array([0, 0, 0, 0]), self.s, AGENT_ID)

    def onPress(self, key) -> None:
        super().onPress(key)
        self.executeAction()

    def onRelease(self, key) -> None:
        super().onRelease(key)
        robot_controller.moveRobot(np.array([0, 0, 0, 0]), self.s, AGENT_ID)


manual_controller = ManualController(OMNI_SPEED, ROTATION_SPEED, LU_SPEED)


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
