import sys
from nat_net_client import NatNetClient
import numpy as np
import math
import pandas as pd
import globals_
import matplotlib.pyplot as plt

m_pos = ['marker_x', 'marker_y', 'marker_z']
rb_pos = ['x', 'y', 'z']
rb_params = ['a', 'b', 'c', 'd']
rb_angles = ['roll', 'pitch', 'yaw']


def _unpack_data(mocap_data):

    markers = []
    rigid_bodies = []

    rigid_body_data = mocap_data.rigid_body_data
    labeled_marker_data = mocap_data.labeled_marker_data

    if labeled_marker_data.get_labeled_marker_count() == 9:
        labeled_marker_list = labeled_marker_data.labeled_marker_list
        rigid_body_list = rigid_body_data.rigid_body_list

        for marker in labeled_marker_list:
            model_id, marker_id = marker.get_id()
            marker = {'model_id': model_id, 'marker_id': marker_id,
                      'marker_x': marker.pos[0], 'marker_y': marker.pos[1], 'marker_z': marker.pos[2]}
            markers.append(marker)

        for rigid_body in rigid_body_list:
            rigid_body = {'id': rigid_body.id_num, 'x': rigid_body.pos[0], 'y': rigid_body.pos[1], 'z': rigid_body.pos[2], 'a': rigid_body.rot[0],
                          'b': rigid_body.rot[1], 'c': rigid_body.rot[2], 'd': rigid_body.rot[3]}
            rigid_bodies.append(rigid_body)

    return markers, rigid_bodies


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


def _rankPoints(markers, head_LU):

    # free_markers = [marker for marker in markers if marker['model_id'] == 0]
    head_LU_pos = np.array([head_LU.get(coord) for coord in rb_pos])

    # Define remaining points and their id's that correspond to thei original indicies
    # remaining_points = [[free_marker.get(coord) for coord in m_pos] for free_marker in free_markers]
    remaining_points = [marker for marker in markers.values() if marker['model_id'] == 0]
    for remaining_point in remaining_points:
        remaining_point['pos'] = np.array([remaining_point.get(coord) for coord in m_pos])
    # remaining_points_ind = list(range(0, 6))

    # List of points ranks
    # ranks = [0] * len(remaining_points)
    rank = 1  # Current rank

    # Find the point closest to the head LU
    distances = np.linalg.norm([point['pos'] for point in remaining_points] - head_LU_pos, axis=1)

    current_index = distances.argmin()

    # Find the rank of the rest of the points
    while remaining_points:
        current_point = remaining_points.pop(current_index)
        markers.get(current_point['marker_id'])['rank'] = rank

        rank += 1  # Update rank

        if(remaining_points):
            distances = np.linalg.norm(
                [point['pos'] for point in remaining_points] - current_point['pos'], axis=1)
            current_index = distances.argmin()

    for marker in markers.values():
        if marker['model_id'] != 0:
            marker['rank'] = 0

    # return markers


def _getAngle(p1, p2):

    a = p2[0] - p1[0]
    b = p2[1] - p1[1]

    alpha = np.arctan2(b, a)

    return alpha


def __calcWheelsCoords(lu1_pos, lu1_angle, lu2_pos, lu2_angle):
    w1_0 = np.array([[-0.022167], [-0.00933]])
    w2_0 = np.array([[0.015833], [-0.036833]])

    w3_0 = np.array([[0.0275], [0]])
    w4_0 = np.array([[-0.0105], [-0.0275]])

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
def getWheelsCoords(markers, rigid_bodies):

    # A list of wheels coordinates
    w = None

    # Retreive location of the markers and rigid_bodies
    # markers, rigid_bodies = _unpack_data(mocap_data)

    if len(markers) == 0 or len(rigid_bodies) == 0:
        raise Exception("No data received from Motive!")
    else:

        # Convert values from the Motive frame to the global frame
        for marker in markers.values():
            new_pos = _motiveToG([marker.get(coord) for coord in m_pos])
            for i in range(3):
                marker[m_pos[i]] = new_pos[i]

        for rigid_body in rigid_bodies.values():
            new_pos = _motiveToG([rigid_body.get(coord) for coord in rb_pos])
            for i in range(3):
                rigid_body[rb_pos[i]] = new_pos[i]

            new_params = _quatTransform(
                [rigid_body.get(param) for param in rb_params])
            for i in range(4):
                rigid_body[rb_params[i]] = new_params[i]

            # Convert quaternions to Euler angles
            euler_angles = _quaternionToEuler(
                [rigid_body.get(param) for param in rb_params])
            for i in range(3):
                rigid_body[rb_angles[i]] = euler_angles[i]

        markers_x = [marker.get('marker_x') for marker in markers.values()]
        markers_y = [marker.get('marker_y') for marker in markers.values()]

        plt.scatter(markers_x, markers_y)

        # Get the position of the head LU
        head_LU = rigid_bodies[1]

        plt.plot(head_LU['x'], head_LU['y'], 'r*')
        plt.plot([head_LU['x'], head_LU['x'] + 0.05 * np.cos(head_LU['yaw'])],
                [head_LU['y'], head_LU['y'] + 0.05 * np.sin(head_LU['yaw'])], 'r')
        
        # Sort markers along the bridge
        _rankPoints(markers, head_LU)

        vsf_markers = [marker for marker in markers.values() if marker['model_id'] == 0 and marker['rank'] != 6]
        vsf_markers.sort(key=lambda marker: marker['rank'])

        vsf_markers_x = [vsf_marker['marker_x'] for vsf_marker in vsf_markers]
        vsf_markers_y = [vsf_marker['marker_y'] for vsf_marker in vsf_markers]

        plt.plot(vsf_markers_x, vsf_markers_y, color='orange')

        plt.axis('equal')
        plt.show()



  
    #     # Position of the tail LU is defined by the last marker in order
    #     lu2_df = markers_df[(markers_df["model_id"] == 0)
    #                         & (markers_df["order"] == 6)]
    #     lu2_pos = [lu2_df.marker_x.values[0], lu2_df.marker_y.values[0]]

    #     # Get the orientation of the head LU
    #     lu1_angle = rb_df[rb_df["id"] == 1].yaw.values[0]

    #     # Get the orientation of the tail LU
    #     p = []
    #     for i in range(3, 6):
    #         p.append(markers_df[markers_df["order"] == i]
    #                  [["marker_x", "marker_y", "marker_z"]].values[0])

    #     alpha1 = _getAngle(p[0], p[1])
    #     alpha2 = _getAngle(p[1], p[2])
    #     lu2_angle = 2 * alpha2 - alpha1

    #     # Calculate the wheels' coordinates from LU's coordinates
    #     w = __calcWheelsCoords(lu1_pos, lu1_angle, lu2_pos, lu2_angle)

    return w
