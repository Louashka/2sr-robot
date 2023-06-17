import main_controller
import motive_client
from pynput import keyboard
import numpy as np
import sys
from NatNetClient import NatNetClient

port_name = "COM3"
controller = mainController.Controller(port_name)

ROTATION_LEFT = {keyboard.KeyCode.from_char('r'), keyboard.Key.left}
ROTATION_RIGHT = {keyboard.KeyCode.from_char('r'), keyboard.Key.right}

LEFT_LU_FORWARD = {keyboard.KeyCode.from_char('1'), keyboard.Key.up}
LEFT_LU_BACKWARD = {keyboard.KeyCode.from_char('1'), keyboard.Key.down}

RIGHT_LU_FORWARD = {keyboard.KeyCode.from_char('2'), keyboard.Key.up}
RIGHT_LU_BACKWARD = {keyboard.KeyCode.from_char('2'), keyboard.Key.down}
BOTH_LU_FORWARD = {keyboard.KeyCode.from_char('3'), keyboard.Key.up}
BOTH_LU_BACKWARD = {keyboard.KeyCode.from_char('3'), keyboard.Key.down}

S1_SOFT = {keyboard.Key.left, keyboard.KeyCode.from_char('s')}
S1_RIGID = {keyboard.Key.left, keyboard.KeyCode.from_char('x')}
S2_SOFT = {keyboard.Key.right, keyboard.KeyCode.from_char('s')}
S2_RIGID = {keyboard.Key.right, keyboard.KeyCode.from_char('x')}
BOTH_SOFT = {keyboard.Key.up, keyboard.KeyCode.from_char('s')}
BOTH_RIGID = {keyboard.Key.down, keyboard.KeyCode.from_char('x')}

# The currently active modifiers
current_keys = set()
s = [0] * 2
data = None


def manual_control(v, s, agent_id):

    while True:
        w_current = motive_client.get_wheels_coords(data)

        if w_current is not None:

            omega = controller.wheel_drive(w_current, v, s)
            controller.move_robot(omega, s, agent_id)
            # controller.move_robot(np.array([0, 0, 0, 0]), s, agent_id)

            break


def on_press(key):
    global current_keys, s

    flag = True

    v = [0] * 5

    omni_speed = 0.1
    rotation_speed = 0.7
    lu_speed = 0.1

    if key in ROTATION_LEFT:
        current_keys.add(key)
        if all(k in current_keys for k in ROTATION_LEFT):
            print('Turn left')
            flag = False
            v[4] = rotation_speed

    if key in ROTATION_RIGHT:
        current_keys.add(key)
        if all(k in current_keys for k in ROTATION_RIGHT):
            print('Turn right')
            flag = False
            v[4] = -rotation_speed

    if key in LEFT_LU_FORWARD:
        current_keys.add(key)
        if all(k in current_keys for k in LEFT_LU_FORWARD):
            print("LU1 forward")
            flag = False
            v[0] = lu_speed

    if key in LEFT_LU_BACKWARD:
        current_keys.add(key)
        if all(k in current_keys for k in LEFT_LU_BACKWARD):
            print("LU1 backward")
            flag = False
            v[0] = -lu_speed

    if key in RIGHT_LU_FORWARD:
        current_keys.add(key)
        if all(k in current_keys for k in RIGHT_LU_FORWARD):
            print("LU2 forward")
            flag = False
            v[1] = lu_speed

    if key in RIGHT_LU_BACKWARD:
        current_keys.add(key)
        if all(k in current_keys for k in RIGHT_LU_BACKWARD):
            print("LU2 backward")
            flag = False
            v[1] = -lu_speed

    if key in BOTH_LU_FORWARD:
        current_keys.add(key)
        if all(k in current_keys for k in BOTH_LU_FORWARD):
            print("LUs forward")
            flag = False
            v[0] = lu_speed
            v[1] = lu_speed

    if key in BOTH_LU_BACKWARD:
        current_keys.add(key)
        if all(k in current_keys for k in BOTH_LU_BACKWARD):
            print("LUs backward")
            flag = False
            v[0] = -lu_speed
            v[1] = -lu_speed

    if key in S1_SOFT:
        current_keys.add(key)
        if all(k in current_keys for k in S1_SOFT):
            print("Segment 1 soft")
            flag = False
            s[0] = 1

    if key in S1_RIGID:
        current_keys.add(key)
        if all(k in current_keys for k in S1_RIGID):
            print("Segment 1 rigid")
            flag = False
            s[0] = 0

    if key in S2_SOFT:
        current_keys.add(key)
        if all(k in current_keys for k in S2_SOFT):
            print("Segment 2 soft")
            flag = False
            s[1] = 1

    if key in S2_RIGID:
        current_keys.add(key)
        if all(k in current_keys for k in S2_RIGID):
            print("Segment 2 rigid")
            flag = False
            s[1] = 0
    if key in BOTH_SOFT:
        current_keys.add(key)
        if all(k in current_keys for k in BOTH_SOFT):
            print("Both segments soft")
            flag = False
            s = [1, 1]

    if key in BOTH_RIGID:
        current_keys.add(key)
        if all(k in current_keys for k in BOTH_RIGID):
            print("Both segments rigid")
            flag = False
            s = [0, 0]

    if flag:

        if key == keyboard.Key.up:
            print("forward")
            v[3] = omni_speed

        if key == keyboard.Key.down:
            print("backward")
            v[3] = -omni_speed

        if key == keyboard.Key.left:
            print("left")
            v[2] = -omni_speed

        if key == keyboard.Key.right:
            print("right")
            v[2] = omni_speed

    manual_control(v, s, 2)


def on_release(key):
    global current_keys
    try:
        current_keys.remove(key)
        controller.move_robot(np.array([0, 0, 0, 0]), s, 2)
    except KeyError:
        pass


def receive_mocap_data_frame(mocap_data):
    global data

    data = mocap_data


def my_parse_args(arg_list, args_dict):
    # set up base values
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

    optionsDict = {}
    optionsDict["clientAddress"] = "127.0.0.1"
    optionsDict["serverAddress"] = "127.0.0.1"
    optionsDict["use_multicast"] = True

    # This will create a new NatNet client
    optionsDict = my_parse_args(sys.argv, optionsDict)

    streaming_client = NatNetClient()
    streaming_client.set_client_address(optionsDict["clientAddress"])
    streaming_client.set_server_address(optionsDict["serverAddress"])
    streaming_client.set_use_multicast(optionsDict["use_multicast"])

    # Configure the streaming client to call our rigid body handler on the emulator to send data out.
    # streaming_client.new_frame_listener = receive_new_frame
    # streaming_client.rigid_body_listener = receive_rigid_body_frame
    streaming_client.mocap_data_listener = receive_mocap_data_frame

    # Start up the streaming client now that the callbacks are set up.
    # This will run perpetually, and operate on a separate thread.
    is_running = streaming_client.run()

    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()
