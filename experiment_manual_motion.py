import robot_controller
import robot_keyboard
import motive_client
from pynput import keyboard
import numpy as np
import globals_
from nat_net_client import NatNetClient

# Constants
OMNI_SPEED = 0.12
ROTATION_SPEED = 1
LU_SPEED = 0.12

AGENT_ID = 1


class ManualController(robot_keyboard.ActionsHandler):

    def __init__(self, omni_speed, rotation_speed, lu_speed) -> None:
        super().__init__(omni_speed, rotation_speed, lu_speed)

        self.__mocap_data = None

    @property
    def mocap_data(self):
        return self.__mocap_data

    @mocap_data.setter
    def mocap_data(self, value) -> None:
        self.__mocap_data = mocap_data

    def executeAction(self):
        try:
            w_current = motive_client.getWheelsCoords(self.mocap_data)

            if w_current is not None:
                omega = robot_controller.getOmega(
                    w_current, self.v, self.s)
                robot_controller.moveRobot(omega, self.s, AGENT_ID)
        except Exception as e:
            # print(f"Error occurred: {e}. The robot is stopped")
            # robot_controller.moveRobot(
            #     np.array([0, 0, 0, 0]), self.s, AGENT_ID)

            w1_0 = 2 * np.array([[-0.005], [-0.0325], [globals_.BETA[0]]])
            w2_0 = 2 * np.array([[0.0325], [0.0045], [globals_.BETA[1]]])

            w3_0 = 2 * np.array([[-0.027], [0.01], [globals_.BETA[2]]])
            w4_0 = 2 * np.array([[0.0105], [-0.027], [globals_.BETA[3]]])

            w_current = [w1_0.T[0], w2_0.T[0], w3_0.T[0], w4_0.T[0]]
            omega = robot_controller.getOmega(
                w_current, self.v, self.s)
            robot_controller.moveRobot(omega, self.s, AGENT_ID)

    def onPress(self, key) -> None:
        super().onPress(key)
        self.executeAction()

    def onRelease(self, key) -> None:
        super().onRelease(key)
        robot_controller.moveRobot(np.array([0, 0, 0, 0]), self.s, AGENT_ID)


manual_controller = ManualController(OMNI_SPEED, ROTATION_SPEED, LU_SPEED)


def receiveMocapDataFrame(data):
    manual_controller.mocap_data(data)


def parseArgs(arg_list, args_dict):
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

    with keyboard.Listener(on_press=manual_controller.onPress, on_release=manual_controller.onRelease) as listener:
        listener.join()
