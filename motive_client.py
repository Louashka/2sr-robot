import sys
from nat_net_client import NatNetClient
import numpy as np
import math
import pandas as pd
import globals_
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

m_pos = ['marker_x', 'marker_y', 'marker_z']
rb_pos = ['x', 'y', 'z']
rb_params = ['a', 'b', 'c', 'd']
rb_angles = ['roll', 'pitch', 'yaw']

# CONSTANTS
# Length and width of the LU  
a = 0.042 
# Distance between LU center and its corner
r = a * math.sqrt(2) / 2 
# Angle between LU orientation and r
alpha = math.radians(-135)

# Coords of the real LU center w.r.t. the rb position
LU_head_center_r = 0.01074968
LU_head_center_angle = math.radians(-60.2551187)


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

    head_LU_pos = np.array([head_LU.get(coord) for coord in rb_pos])

    # Define remaining points and their id's that correspond to thei original indicies
    remaining_points = [marker for marker in markers.values() if marker['model_id'] == 0]
    for remaining_point in remaining_points:
        remaining_point['pos'] = np.array([remaining_point.get(coord) for coord in m_pos])

    # List of points ranks
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


def _calcLUHeadOrientation(LU_head_markers):
    points = []

    for marker in LU_head_markers:
        points.append([marker['marker_x'], marker['marker_y']])

    points = np.array(points)
    triangle_sides = np.linalg.norm(points - np.roll(points, -1, 0), axis=1)

    i = triangle_sides.argsort()[1]
    i_next = i + 1 if i < 2 else 0

    delta = points[i, :] - points[i_next, :]
    theta = np.arctan2(delta[1], delta[0]) + math.pi

    return theta

def _getAngle(p1, p2):

    a = p2[0] - p1[0]
    b = p2[1] - p1[1]

    alpha = np.arctan2(b, a)

    return alpha


def __calcWheelsCoords(LU_head_frame, LU_tail_frame):
    w1_0 = np.array([[-0.0275], [0]])
    w2_0 = np.array([[0.0105], [-0.0275]])

    w3_0 = np.array([[0.0275], [0]])
    w4_0 = np.array([[-0.0105], [-0.027]])

    R1 = np.array([[np.cos(LU_head_frame[2]), -np.sin(LU_head_frame[2])],
                   [np.sin(LU_head_frame[2]), np.cos(LU_head_frame[2])]])
    w1 = np.matmul(R1, w1_0).T[0] + LU_head_frame[:2]
    w2 = np.matmul(R1, w2_0).T[0] + LU_head_frame[:2]

    R2 = np.array([[np.cos(LU_tail_frame[2]), -np.sin(LU_tail_frame[2])],
                   [np.sin(LU_tail_frame[2]), np.cos(LU_tail_frame[2])]])
    w3 = np.matmul(R2, w3_0).T[0] + LU_tail_frame[:2]
    w4 = np.matmul(R2, w4_0).T[0] + LU_tail_frame[:2]

    w = [w1, w2, w3, w4]

    return w

def _wheelsToBodyFrame(body_frame, LU_head, LU_tail_angle, w):

    R_origin = np.array([[np.cos(body_frame[2]), -np.sin(body_frame[2])],
                         [np.sin(body_frame[2]), np.cos(body_frame[2])]])

    for i in range(4):
        w_b0 = np.array([w[i] - body_frame[:2]]).T
        w[i] = np.matmul(R_origin, w_b0).T[0]

    for i in range(2):
        w[i] = np.append(w[i], LU_head['yaw'] + globals_.BETA[i])

    for i in range(2, 4):
        w[i] = np.append(w[i], LU_tail_angle + globals_.BETA[i])

    return w

def _displayRobot(markers, all_frames, wheels):

    # Unpack frames
    LU_head_frame, LU_tail_frame, body_frame = all_frames

    fig, ax = plt.subplots()

    # Plot all captured markers
    markers_x = [marker.get('marker_x') for marker in markers.values()]
    markers_y = [marker.get('marker_y') for marker in markers.values()]

    ax.scatter(markers_x, markers_y)
    
    # Plot the block of the head LU
    LU_head_rect = (LU_head_frame[0] + r*np.cos(LU_head_frame[2] + alpha), LU_head_frame[1] + r*np.sin(LU_head_frame[2] + alpha))
    ax.add_patch(Rectangle(LU_head_rect, a, a, angle=math.degrees(LU_head_frame[2]), edgecolor='black', facecolor='none'))

    # Plot the block of the tail LU
    LU_tail_centre = (LU_tail_frame[0] + r*np.cos(LU_tail_frame[2] + alpha), LU_tail_frame[1] + r*np.sin(LU_tail_frame[2] + alpha))
    ax.add_patch(Rectangle(LU_tail_centre, a, a, angle=math.degrees(LU_tail_frame[2]), edgecolor='black', facecolor='none'))

    # Plot all frames
    for frame in all_frames:
        ax.plot(frame[0], frame[1], 'r*')
        ax.plot([frame[0], frame[0] + 0.03 * np.cos(frame[2])], [frame[1], frame[1] + 0.03 * np.sin(frame[2])], 'r')

    # Plot wheels
    for wheel in wheels:
        ax.plot(wheel[0], wheel[1], 'mo', markersize=15)

    # Plot the bridge curve
    vsf_markers = [marker for marker in markers.values() if marker['model_id'] == 0 and marker['rank'] != 6]
    vsf_markers.sort(key=lambda marker: marker['rank'])

    vsf_markers_x = [vsf_marker['marker_x'] for vsf_marker in vsf_markers]
    vsf_markers_y = [vsf_marker['marker_y'] for vsf_marker in vsf_markers]

    ax.plot(vsf_markers_x, vsf_markers_y, color='orange')

    # Connect the bridge with the LU's
    vsf_start = (LU_head_frame[0] + r*np.cos(LU_head_frame[2] + alpha - np.pi), LU_head_frame[1] + r*np.sin(LU_head_frame[2] + alpha - np.pi))
    ax.plot([vsf_start[0], vsf_markers_x[0]], [vsf_start[1], vsf_markers_y[0]], color='orange')

    vsf_end = (LU_tail_frame[0] + r*np.cos(LU_tail_frame[2] + alpha - np.pi/2), LU_tail_frame[1] + r*np.sin(LU_tail_frame[2] + alpha - np.pi/2))
    ax.plot([vsf_end[0], vsf_markers_x[-1]], [vsf_end[1], vsf_markers_y[-1]], color='orange')

    ax.axis('equal')
    plt.show()


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

        
        # Get position of the head LU
        LU_head_rb = rigid_bodies[1]
        # Rank markers along the bridge
        _rankPoints(markers, LU_head_rb)

        # Define the frame of the head LU
        LU_head_theta = _calcLUHeadOrientation([marker for marker in markers.values() if marker['model_id'] == 1])
        LU_head_x = LU_head_rb['x'] + LU_head_center_r*np.cos(LU_head_theta + LU_head_center_angle)
        LU_head_y = LU_head_rb['y'] + LU_head_center_r*np.sin(LU_head_theta + LU_head_center_angle)

        LU_head_frame = [LU_head_x, LU_head_y, LU_head_theta]

        # Position of the tail LU is defined by the last marker in order
        LU_tail = [marker for marker in markers.values() if marker['model_id'] == 0 and marker['rank'] == 6][0]

        # Get the orientation of the tail LU
        p = []
        for i in range(3, 6):
            for marker in markers.values():
                if marker['rank'] == i:
                    p.append(marker['pos'])

        alpha1 = _getAngle(p[0], p[1])
        alpha2 = _getAngle(p[1], p[2])
        LU_tail_angle = 2 * alpha2 - alpha1

        # Define the frame of the tail LU
        LU_tail_frame = [LU_tail['marker_x'], LU_tail['marker_y'], LU_tail_angle]

        # Define the body frame
        body_frame = [(LU_head_x + LU_tail_frame[0]) / 2, (LU_head_y + LU_tail_frame[1]) / 2]
        body_frame_theta = _getAngle(LU_head_frame[:2],LU_tail_frame[:2])

        body_frame.append(body_frame_theta)

        # Combine all frames
        all_frames = [LU_head_frame, LU_tail_frame, body_frame]

        # Calculate the wheels' coordinates from LU's coordinates
        wheels = __calcWheelsCoords(LU_head_frame, LU_tail_frame)

        # Plot a robot 
        _displayRobot(markers, all_frames, wheels)

    return wheels
