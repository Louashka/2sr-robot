import sys
from nat_net_client import NatNetClient
import numpy as np
import pandas as pd
import globals_


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

    if labeled_marker_data.get_labeled_marker_count() == 9:
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


def _motiveToG(coords):
    R_motive_to_g = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])

    return np.matmul(R_motive_to_g, np.array(coords)).tolist()


def _quatTransform(args):
    return [-args[0], args[2], args[1], args[3]]


def _quaternionToEuler(coeffs):
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


def _rankPoints(markers_df, lu1_df):

    free_markers_df = markers_df[markers_df['model_id'] == 0]

    # Define remaining points and their id's that correspond to thei original indicies
    remaining_points = free_markers_df[[
        'marker_x', 'marker_y', 'marker_z']].values.tolist()
    remaining_points_id = list(range(0, 6))

    # List of points ranks
    ranks = [0] * len(remaining_points)
    rank = 1  # Current rank

    # Find the point closest to the head LU
    distances = np.linalg.norm(
        remaining_points - lu1_df[['x', 'y', 'z']].values, axis=1)

    current_index = distances.argmin()
    current_point_id = remaining_points_id[current_index]

    # Find the rank of the rest of the points
    while remaining_points:
        current_point = remaining_points.pop(current_index)
        current_point_id = remaining_points_id.pop(current_index)

        ranks[current_point_id] = rank
        rank += 1  # Update rank

        if(remaining_points):
            distances = np.linalg.norm(
                remaining_points - np.array(current_point), axis=1)
            current_index = distances.argmin()

    markers_df['order'] = 0
    markers_df['order'][markers_df['model_id'] == 0] = ranks

    # Sort the 'markers_df' DataFrame by order
    sorted_markers_df = markers_df.sort_values(by=['order'])

    return sorted_markers_df


def _getAngle(p1, p2):

    a = p2[0] - p1[0]
    b = p2[1] - p1[1]

    alpha = np.arctan2(b, a)

    return alpha


def __calcWheelsCoords(lu1_pos, lu1_angle, lu2_pos, lu2_angle):
    w1_0 = 2 * np.array([[-0.005], [-0.0325]])
    w2_0 = 2 * np.array([[0.0325], [0.0045]])

    w3_0 = 2 * np.array([[-0.027], [0.01]])
    w4_0 = 2 * np.array([[0.0105], [-0.027]])

    R1 = np.array([[np.cos(lu1_angle), -np.sin(lu1_angle)],
                   [np.sin(lu1_angle), np.cos(lu1_angle)]])
    w1 = np.matmul(R1, w1_0).T[0] + lu1_pos
    w2 = np.matmul(R1, w2_0).T[0] + lu1_pos

    R2 = np.array([[np.cos(lu2_angle), -np.sin(lu2_angle)],
                   [np.sin(lu2_angle), np.cos(lu2_angle)]])
    w3 = np.matmul(R2, w3_0).T[0] + lu2_pos
    w4 = np.matmul(R2, w4_0).T[0] + lu2_pos

    w = [w1, w2, w3, w4]

    centroid = [(lu1_pos[0] + lu2_pos[0]) / 2, (lu1_pos[1] + lu2_pos[1]) / 2]
    direction = _get_angle(lu1_pos, lu2_pos)

    R_origin = np.array([[np.cos(direction), -np.sin(direction)],
                         [np.sin(direction), np.cos(direction)]])

    for i in range(4):
        w_b0 = np.array([w[i] - centroid]).T
        w[i] = np.matmul(R_origin, w_b0).T[0]

    for i in range(2):
        w[i] = np.append(w[i], lu1_angle + globals_.BETA[i])

    for i in range(2, 4):
        w[i] = np.append(w[i], lu2_angle + globals_.BETA[i])

    return w


# Calculate the wheels' coordinates from the mocap data
def getWheelsCoords(mocap_data):

    # A list of wheels coordinates
    w = None

    # Retreive location of the markers and rigid_bodies
    markers_df, rb_df = _unpack_data(mocap_data)

    if markers_df.empty or rb_df.empty:
        raise Exception("No data received from Motive!")
    else:
        # Convert values from the Motive frame to the global frame
        markers_df[['marker_x', 'marker_y', 'marker_z']] = markers_df[[
            'marker_x', 'marker_y', 'marker_z']].apply(lambda x: pd.Series(_motiveToG(x)), axis=1)
        rb_df[['x', 'y', 'z']] = rb_df[['x', 'y', 'z']].apply(lambda x: pd.Series(
            [_motiveToG(x)[0], _motiveToG(x)[1], _motiveToG(x)[2]]), axis=1)
        rb_df[['a', 'b', 'c', 'd']] = rb_df[['a', 'b', 'c', 'd']].apply(
            lambda x: pd.Series(_quatTransform(x)), axis=1)

        # Convert quaternions to Euler angles
        rb_df[['roll', 'pitch', 'yaw']] = rb_df[['a', 'b', 'c', 'd']].apply(
            lambda x: pd.Series(_quaternionToEuler(x)), axis=1)

        # Sort markers along the bridge
        markers_df = _sortPoints(markers_df, rb_df)

        # Get the position of the locomotion units
        lu1_pos = [rb_df[rb_df["id"] == 1].x.values[0],
                   rb_df[rb_df["id"] == 1].y.values[0]]

        # Position of the tail LU is defined by the last marker in order
        lu2_df = markers_df[(markers_df["model_id"] == 0)
                            & (markers_df["order"] == 6)]
        lu2_pos = [lu2_df.marker_x.values[0], lu2_df.marker_y.values[0]]

        # Get the orientation of the head LU
        lu1_angle = rb_df[rb_df["id"] == 1].yaw.values[0]

        # Get the position of the tail LU
        p = []
        for i in range(3, 6):
            p.append(markers_df[markers_df["order"] == i]
                     [["marker_x", "marker_y", "marker_z"]].values[0])

        alpha1 = _getAngle(p[0], p[1])
        alpha2 = _getAngle(p[1], p[2])
        lu2_angle = 2 * alpha2 - alpha1

        # Calculate the wheels' coordinates from LU's coordinates
        w = __calcWheelsCoords(lu1_pos, lu1_angle, lu2_pos, lu2_angle)

    return w
