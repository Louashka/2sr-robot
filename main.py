# import robot_controller
import Controller.keyboard_controller as keyboard_controller
import Motive.motive_client as motive_client
from enum import Enum
import numpy as np
import pandas as pd
import math
from Motive.nat_net_client import NatNetClient
import sys

class ExpMode(Enum):
    MANUAL = 1
    COLLAB = 2
    COOP = 3

class Experiment:
    def __init__(self) -> None:
        self.__is_running = False
        self.__mocapData = None

    def __receiveMocapData(self, value) -> None:
        self.__mocapData = value

    def __parseArgs(self, arg_list, args_dict):
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
    
    def startDataListener(self):
        options_dict = {}
        options_dict["clientAddress"] = "127.0.0.1"
        options_dict["serverAddress"] = "127.0.0.1"
        options_dict["use_multicast"] = True

        # Start Motive streaning
        options_dict = self.__parseArgs(sys.argv, options_dict)

        streaming_client = NatNetClient()
        streaming_client.set_client_address(options_dict["clientAddress"])
        streaming_client.set_server_address(options_dict["serverAddress"])
        streaming_client.set_use_multicast(options_dict["use_multicast"])

        streaming_client.mocap_data_listener = self.__receiveMocapData

        self.__is_running = streaming_client.run()

    def run(self, mode):
        if not self.__is_running:
            raise Exception('Please start the streaming!')
        
        print('Start the experiment')
        
        match mode:
            case ExpMode.MANUAL:
                print('Manual mode')
            case ExpMode.COLLAB:
                print('Collaboration mode')
            case ExpMode.COOP:
                print('Cooperation mode')



if __name__ == "__main__":
    experiment = Experiment()
    experiment.startDataListener()
    experiment.run(ExpMode.COLLAB)