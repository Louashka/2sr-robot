import sys
from nat_net_client import NatNetClient
import math
import numpy as np
import scipy.optimize as so
import pandas as pd

count = 0

data_model_id = []
data_marker_id = []
data_marker_x = []
data_marker_y = []
data_marker_z = []

data_rb_id = []
data_rb_x = []
data_rb_y = []
data_rb_z = []
data_rb_rot_a = []
data_rb_rot_b = []
data_rb_rot_c = []
data_rb_rot_d = []


def receive_new_frame(data_dict):
    order_list = ["frameNumber", "markerSetCount", "unlabeledMarkersCount", "rigidBodyCount", "skeletonCount",
                  "labeledMarkerCount", "timecode", "timecodeSub", "timestamp", "isRecording", "trackedModelsChanged"]
    dump_args = False
    if dump_args == True:
        out_string = "    "
        for key in data_dict:
            out_string += key + "="
            if key in data_dict:
                out_string += data_dict[key] + " "
            out_string += "/"


def receive_rigid_body_frame(new_id, position, rotation):
    pass
    #print( "Received frame for rigid body", new_id )
    # print("Received frame for rigid body",
    # new_id, " ", position, " ", rotation)


def euler_from_quaternion(rot):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    x, y, z, w = rot
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    return roll_x, pitch_y, yaw_z  # in radians


def transform_from_motive_frame(var, c_type="pos"):
    R = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])
    if c_type == "pos":
        return np.matmul(R, np.array(var))
    elif c_type == "ang":
        return np.array([-var[0], var[2], var[1]])
    else:
        return "Error"


def fun1(k, *args):
    l, l0, th0, p1, p0 = args
    # print("l: %s, l0: %s, th0: %s, p1: %s, p0: %s" % (l, l0, th0, p1, p0))
    th = th0 + k * l
    eq1 = p0[0] - l0 / 2 * np.cos(th) - p1[0] - \
        np.sin(th) / k + np.sin(th0) / k
    eq2 = p0[1] - l0 / 2 * np.sin(th) - p1[1] + \
        np.cos(th) / k - np.cos(th0) / k
    # return [eq1, eq2]
    return eq1


def fun2(k, *args):
    l, l0, phi, p1, p0, cup = args
    # print("l: %s, l0: %s, th0: %s, p1: %s, p0: %s" % (l, l0, th0, p1, p0))
    th = phi + k * l
    eq1 = p1[0] + np.sin(th) / k - np.sin(phi) / k - \
        p0[0] + np.cos(th) * cup[0] + np.sin(th) * cup[1]
    return eq1


def receive_mocap_data_frame(mocap_data):
    global count, data_model_id, data_marker_id, data_marker_x, data_marker_y, data_marker_z, data_rb_id, data_rb_x, data_rb_y, data_rb_z, data_rb_rot_a, data_rb_rot_b, data_rb_rot_c, data_rb_rot_d

    # print("%s\n"%mocap_data.labeled_marker_data.get_as_string(" ", 1))

    if count == 1:
        data1 = {'pose': 1, 'model_id': data_model_id, 'marker_id': data_marker_id,
                 'marker_x': data_marker_x, 'marker_y': data_marker_y, 'marker_z': data_marker_z}
        df1 = pd.DataFrame(data=data1)
        df1.to_csv('markers.csv', index=False, mode='a', header=False)

        data2 = {'pose': 1, 'id': data_rb_id, 'x': data_rb_x, 'y': data_rb_y, 'z': data_rb_z,
                 'a': data_rb_rot_a, 'b': data_rb_rot_b, 'c': data_rb_rot_c, 'd': data_rb_rot_d}
        df2 = pd.DataFrame(data=data2)
        df2.to_csv('rigid_bodies.csv', index=False, mode='a', header=False)

    agent1_config = [0, 0, 0, 0, 0]
    LU1_a = 0
    LU1_c = 0
    seg1_cup = (35 * 2) / 1000
    l = 0.08
    l0 = 0.046
    rigid_body_data = mocap_data.rigid_body_data
    labeled_marker_data = mocap_data.labeled_marker_data

    if labeled_marker_data.get_labeled_marker_count() == 12:
        rigid_body_list = rigid_body_data.rigid_body_list
        labeled_marker_list = labeled_marker_data.labeled_marker_list

        LU1_markers = {}
        labeled_markers = []

        for marker in labeled_marker_list:
            model_id, marker_id = marker.get_id()

            if model_id == 1:
                LU1_markers[marker_id] = transform_from_motive_frame(
                    marker.pos, c_type="pos")
            if model_id == 0:
                labeled_markers.append(
                    transform_from_motive_frame(marker.pos, c_type="pos"))

            data_model_id.append(model_id)
            data_marker_id.append(marker_id)
            data_marker_x.append(marker.pos[0])
            data_marker_y.append(marker.pos[1])
            data_marker_z.append(marker.pos[2])

        LU1_a = np.linalg.norm(
            np.array(LU1_markers[1]) - np.array(LU1_markers[2]))
        LU1_c = np.linalg.norm(
            np.array(LU1_markers[1]) - np.array(LU1_markers[3]))
        LU1_h = LU1_c / 2
        LU1_pos = 0.5 * LU1_h * \
            np.array([np.cos(-np.pi / 4), np.sin(-np.pi / 4)])
        VSS_cup_coord = np.array([LU1_a / 2 + seg1_cup, LU1_a / 2])
        VSS_cup_coord_ = np.array([LU1_a / 2 + seg1_cup, -LU1_a / 2])

        if labeled_markers[0][1] < labeled_markers[1][1]:
            origin = labeled_markers[0]
            LU2 = labeled_markers[1]
        else:
            origin = labeled_markers[1]
            LU2 = labeled_markers[0]

        agent1_config[0] = origin[0]
        agent1_config[1] = origin[1]

        print(f"Origin pos x: {origin[0]}")
        print(f"LU2 pos x: {LU2[0]}")

        for rigid_body in rigid_body_list:
            if rigid_body.id_num == 1:
                rb_pos = transform_from_motive_frame(
                    rigid_body.pos, c_type="ang")
                motive_angles = euler_from_quaternion(rigid_body.rot)
                LU1_orient = transform_from_motive_frame(
                    motive_angles, c_type="ang")

                R_lu1 = np.array([[np.cos(LU1_orient[2]), -np.sin(LU1_orient[2])],
                                  [np.sin(LU1_orient[2]), np.cos(LU1_orient[2])]])
                rb_pos_2d = np.array(rb_pos[:-1])
                LU1_origin = rb_pos_2d + np.matmul(R_lu1, LU1_pos)
                seg1_start_pos = LU1_origin + np.matmul(R_lu1, VSS_cup_coord)
                k1 = so.fsolve(fun1, 1, args=(
                    l, l0, LU1_orient[2], seg1_start_pos, origin[:-1]))[0]
                agent1_config[2] = LU1_orient[2] + k1 * l
                seg2_start_pos = np.array(
                    origin[-1]) + l0 / 2 * np.array([np.cos(agent1_config[2]), np.sin(agent1_config[2])])
                k2 = so.fsolve(fun2, 1, args=(
                    l, l0, agent1_config[2], seg2_start_pos, LU2[:-1], VSS_cup_coord_))[0]
                print(f"k1: {k1}")
                print(f"k2: {k2}")

            elif rigid_body.id_num == 2:
                print(f"Manipulandum's position: {rigid_body.pos}")

            data_rb_id.append(rigid_body.id_num)
            data_rb_x.append(rigid_body.pos[0])
            data_rb_y.append(rigid_body.pos[1])
            data_rb_z.append(rigid_body.pos[2])
            data_rb_rot_a.append(rigid_body.rot[0])
            data_rb_rot_b.append(rigid_body.rot[1])
            data_rb_rot_c.append(rigid_body.rot[2])
            data_rb_rot_d.append(rigid_body.rot[3])

        count += 1


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
