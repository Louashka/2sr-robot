import nat_net_client as nnc
import sys

data = None


def receiveData(value) -> None:
    global data
    data = value


def parseArgs(arg_list, args_dict) -> dict:
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


def startDataListener() -> None:
    options_dict = {}
    options_dict["clientAddress"] = "127.0.0.1"
    options_dict["serverAddress"] = "127.0.0.1"
    options_dict["use_multicast"] = True

    # Start Motive streaning
    options_dict = parseArgs(sys.argv, options_dict)

    streaming_client = nnc.NatNetClient()
    streaming_client.set_client_address(options_dict["clientAddress"])
    streaming_client.set_server_address(options_dict["serverAddress"])
    streaming_client.set_use_multicast(options_dict["use_multicast"])

    streaming_client.mocap_data_listener = receiveData

    streaming_client.run()


def unpackData():
    if data is not  None:

        labeled_marker_data = data.labeled_marker_data

        labeled_marker_list = labeled_marker_data.labeled_marker_list

        for marker in labeled_marker_list:
            model_id, marker_id = [int(i) for i in marker.get_id()]
            marker = {'model_id': model_id, 'marker_id': marker_id,
                      'marker_x': marker.pos[0], 'marker_y': marker.pos[1], 'marker_z': marker.pos[2]}
            print(marker)

if __name__ == "__main__":
    startDataListener()

    while True:
        unpackData()

