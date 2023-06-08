import sys
from NatNetClient import NatNetClient

optionsDict = {}
optionsDict["clientAddress"] = "127.0.0.1"
optionsDict["serverAddress"] = "127.0.0.1"
optionsDict["use_multicast"] = True

mocap_data = None


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


# This will create a new NatNet client
optionsDict = my_parse_args(sys.argv, optionsDict)

streaming_client = NatNetClient()
streaming_client.set_client_address(optionsDict["clientAddress"])
streaming_client.set_server_address(optionsDict["serverAddress"])
streaming_client.set_use_multicast(optionsDict["use_multicast"])


def get_wheels_coords():
    w_coords = [[0] * 3] * 4

    mocap_data = streaming_client.get_current_frame_data()

    return w_coords
