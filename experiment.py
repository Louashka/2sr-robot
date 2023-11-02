import agent_controller
import keyboard_controller
import motive_client
from pynput import keyboard
import numpy as np
import sys
from nat_net_client import NatNetClient

# Constants
# PORT_NAME = "COM5"

OMNI_SPEED = 0.12
ROTATION_SPEED = 1
LU_SPEED = 0.12

# Create the controller
# controller = agent_controller.Controller(PORT_NAME)
kc = keyboard.Controller(OMNI_SPEED, ROTATION_SPEED, LU_SPEED)

data = None
agent_id = 1

# Manual Control Function


def manualControl(v, s, agent_id):
    try:
        w_current = motive_client.getWheelsCoords(data)

        if w_current is not None:
            omega = controller.getWheelsVelocities(w_current, v, s)
            controller.moveRobot(omega, s, agent_id)
    except Exception as e:
        print(f"Error occurred: {e}. The robot is stopped")
        controller.moveRobot(np.array([0, 0, 0, 0]), s, agent_id)


def onPress(key):
    v, s = kc.onPress()
    manualControl(v, s, agent_id)


def onRelease(key):
    global current_keys
    try:
        current_keys.remove(key)
        controller.moveRobot(np.array([0, 0, 0, 0]), s, agent_id)
    except KeyError:
        pass


def receiveMocapDataFrame(mocap_data):
    global data
    data = mocap_data


def parseArgs(arg_list, args_dict):
    # Implementation of parse_args method
    pass


if __name__ == "__main__":
    # options_dict = {}
    # options_dict["clientAddress"] = "127.0.0.1"
    # options_dict["serverAddress"] = "127.0.0.1"
    # options_dict["use_multicast"] = True

    # # This will create a new NatNet client
    # options_dict = parseArgs(sys.argv, options_dict)

    # streaming_client = NatNetClient()
    # streaming_client.set_client_address(options_dict["clientAddress"])
    # streaming_client.set_server_address(options_dict["serverAddress"])
    # streaming_client.set_use_multicast(options_dict["use_multicast"])

    # streaming_client.mocap_data_listener = receiveMocapDataFrame

    # is_running = streaming_client.run()

    with keyboard.Listener(on_press=onPress, on_release=onRelease) as listener:
        listener.join()
