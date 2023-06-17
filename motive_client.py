import sys
from nat_net_client import NatNetClient
import numpy as np
import pandas as pd

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


def _unpack_data(mocap_data):
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

    rigid_body_data = mocap_data.rigid_body_data
    labeled_marker_data = mocap_data.labeled_marker_data

    if labeled_marker_data.get_labeled_marker_count() == 12:
        rigid_body_list = rigid_body_data.rigid_body_list
        labeled_marker_list = labeled_marker_data.labeled_marker_list

        for marker in labeled_marker_list:
            model_id, marker_id = marker.get_id()

            data_model_id.append(model_id)
            data_marker_id.append(marker_id)
            data_marker_x.append(marker.pos[0])
            data_marker_y.append(marker.pos[1])
            data_marker_z.append(marker.pos[2])

        for rigid_body in rigid_body_list:
            data_rb_id.append(rigid_body.id_num)
            data_rb_x.append(rigid_body.pos[0])
            data_rb_y.append(rigid_body.pos[1])
            data_rb_z.append(rigid_body.pos[2])
            data_rb_rot_a.append(rigid_body.rot[0])
            data_rb_rot_b.append(rigid_body.rot[1])
            data_rb_rot_c.append(rigid_body.rot[2])
            data_rb_rot_d.append(rigid_body.rot[3])

    data1 = {'model_id': data_model_id, 'marker_id': data_marker_id,
             'marker_x': data_marker_x, 'marker_y': data_marker_y, 'marker_z': data_marker_z}
    markers_df = pd.DataFrame(data=data1)

    data2 = {'id': data_rb_id, 'x': data_rb_x, 'y': data_rb_y, 'z': data_rb_z,
             'a': data_rb_rot_a, 'b': data_rb_rot_b, 'c': data_rb_rot_c, 'd': data_rb_rot_d}
    rb_df = pd.DataFrame(data=data2)

    return markers_df, rb_df


def _motive_to_g(coords):
    R_motive_to_g = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])

    return np.matmul(R_motive_to_g, np.array(coords)).tolist()


def _quat_tranform(args):
    return [-args[0], args[2], args[1], args[3]]


def _euler_from_quaternion(coeffs):
    t0 = +2.0 * (coeffs[3] * coeffs[0] + coeffs[1] * coeffs[2])
    t1 = +1.0 - 2.0 * (coeffs[0] * coeffs[0] + coeffs[1] * coeffs[1])
    roll_x = np.arctan2(t0, t1)

    t2 = +2.0 * (coeffs[3] * coeffs[1] - coeffs[2] * coeffs[0])
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = np.arcsin(t2)

    t3 = +2.0 * (coeffs[3] * coeffs[2] + coeffs[0] * coeffs[1])
    t4 = +1.0 - 2.0 * (coeffs[1] * coeffs[1] + coeffs[2] * coeffs[2])
    yaw_z = np.arctan2(t3, t4)

    return [roll_x, pitch_y, yaw_z]  # in radians


def _sort_points(marker_df, rb_df):
    marker_df["order"] = 0

    lu1_df = rb_df[rb_df["id"] == 2]
    free_markers_df = marker_df[df["model_id"] == 0]

    vsf_start = free_markers_df.head(1)
    min_dist = np.linalg.norm(
        vsf_start[['marker_x', 'marker_y', 'marker_z']].values - lu1_df[['x', 'y', 'z']].values)
    min_index = vsf_start.index[0]

    for index, row in free_markers_df.iterrows():
        dist = np.linalg.norm(
            row[['marker_x', 'marker_y', 'marker_z']].values - lu1_df[['x', 'y', 'z']].values)

        if dist < min_dist:
            min_dist = dist
            min_index = index

    markers_df_.at[min_index, 'order'] = 1
    free_markers_pose_df.at[min_index, 'order'] = 1

    for i in range(1, 6):
        current_df = free_markers_df[free_markers_df['order'] == 0]

        current_point = free_markers_df[free_markers_df['order'] == i]
        next_point = current_df.head(1)

        min_dist = np.linalg.norm(next_point[['marker_x', 'marker_y', 'marker_z']
                                             ].values - current_point[['marker_x', 'marker_y', 'marker_z']].values[0])
        min_index = next_point.index[0]

        for index, row in current_df.iterrows():
            dist = np.linalg.norm(row[['marker_x', 'marker_y', 'marker_z']].values -
                                  current_point[['marker_x', 'marker_y', 'marker_z']].values[0])

            if dist < min_dist:
                min_dist = dist
                min_index = index

        markers_df.at[min_index, 'order'] = i + 1
        free_markers_df.at[min_index, 'order'] = i + 1

    return markers_df.sort_values(by=['order'])


def _get_angle(p1, p2):

    a = p2[0] - p1[0]
    b = p2[1] - p1[1]

    alpha = np.arctan2(b, a)

    return alpha


def __calc_wheels_ccoords(lu1_angle, lu2_angle):
    w1_0 = 2 * np.array([[-0.005], [-0.0325]])
    w2_0 = 2 * np.array([[0.0325], [0.0045]])

    w3_0 = 2 * np.array([[-0.027], [0.01]])
    w4_0 = 2 * np.array([[0.0105], [-0.027]])

    R1 = np.array([[np.cos(lu1_angle), -np.sin(lu1_angle)],
                   [np.sin(lu1_angle), np.cos(lu1_angle)]])
    w1 = np.matmul(R1, w1_0)
    w2 = np.matmul(R1, w2_0)

    R2 = np.array([[np.cos(lu2_angle), -np.sin(lu2_angle)],
                   [np.sin(lu2_angle), np.cos(lu2_angle)]])
    w3 = np.matmul(R2, w3_0)
    w4 = np.matmul(R2, w4_0)

    w = [w1, w2, w3, w4]

    return w


def get_wheels_coords():
    w_coords = []

    mocap_data = streaming_client.get_current_frame_data()

    if mocap_data is not None:
        markers_df, rb_df = _unpack_data(mocap_data)

        markers_df[['marker_x', 'marker_y', 'marker_z']] = markers_df_[['marker_x', 'marker_y', 'marker_z']].apply(
            lambda x: pd.Series([_motive_to_g(x)[0], _motive_to_g(x)[1], _motive_to_g(x)[2]]), axis=1)
        rb_df[['x', 'y', 'z']] = rb_df[['x', 'y', 'z']].apply(lambda x: pd.Series(
            [_motive_to_g(x)[0], _motive_to_g(x)[1], _motive_to_g(x)[2]]), axis=1)
        rb_df[['a', 'b', 'c', 'd']] = rb_df[['a', 'b', 'c', 'd']].apply(lambda x: pd.Series(
            [_quat_tranform(x)[0], _quat_tranform(x)[1], _quat_tranform(x)[2], _quat_tranform(x)[3]]), axis=1)

        rb_df[['roll', 'pitch', 'yaw']] = rb_df[['a', 'b', 'c', 'd']].apply(lambda x: pd.Series(
            [_euler_from_quaternion(x)[0], _euler_from_quaternion(x)[1], _euler_from_quaternion(x)[2]]), axis=1)
        markers_df = _sort_points(markers_df, rb_df)

        lu1_angle = rb_df[rb_df["id"] == 2].yaw.values[0]

        p = []
        for i in range(3, 6):
            p.append(markers_df[markers_df["order"] == i]
                     [["marker_x", "marker_y", "marker_z"]].values[0])

        alpha1 = _get_angle(p[0], p[1])
        alpha2 = _get_angle(p[1], p[2])
        lu2_angle = 2 * alpha2 - alpha1

        w_coords = __calc_wheels_ccoords(lu1_angle, lu2_angle)

    return w_coords
