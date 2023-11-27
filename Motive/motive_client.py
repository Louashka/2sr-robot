from Motive.nat_net_client import NatNetClient
import numpy as np
import math
import copy
import Model.agent as agent
import Model.global_var as global_var

m_pos = ['marker_x', 'marker_y', 'marker_z']
rb_pos = ['x', 'y', 'z']
rb_params = ['a', 'b', 'c', 'd']
rb_angles = ['roll', 'pitch', 'yaw']

# Coords of the real LU center w.r.t. the rb position
LU_head_center_r = 0.01074968
LU_head_center_angle = math.radians(-60.2551187)

def _unpackData(mocap_data):

    rigid_body_data = mocap_data.rigid_body_data
    labeled_marker_data = mocap_data.labeled_marker_data

    labeled_marker_list = labeled_marker_data.labeled_marker_list
    rigid_body_list = rigid_body_data.rigid_body_list

    markers = {}
    rigid_bodies = {}

    for marker in labeled_marker_list:
        model_id, marker_id = [int(i) for i in marker.get_id()] 
        marker = {'model_id': model_id, 'marker_id': marker_id,
                    'marker_x': marker.pos[0], 'marker_y': marker.pos[1], 'marker_z': marker.pos[2]}
        markers[str(model_id) + '.' + str(marker_id)] = marker

    for rigid_body in rigid_body_list:
        rigid_body = {'id': int(rigid_body.id_num), 'x': rigid_body.pos[0], 'y': rigid_body.pos[1], 'z': rigid_body.pos[2], 'a': rigid_body.rot[0],
                        'b': rigid_body.rot[1], 'c': rigid_body.rot[2], 'd': rigid_body.rot[3]}
        rigid_bodies[int(rigid_body['id'])] = rigid_body    

    return markers, rigid_bodies

def _convertData(markers, rigid_bodies):
    # Convert values from the Motive frame to the global frame
    for marker in markers.values():
        new_pos = _positionToGlobal([marker.get(coord) for coord in m_pos])
        for i in range(3):
            marker[m_pos[i]] = new_pos[i]

    for rigid_body in rigid_bodies.values():
        new_pos = _positionToGlobal([rigid_body.get(coord) for coord in rb_pos])
        for i in range(3):
            rigid_body[rb_pos[i]] = new_pos[i]

        new_params = _quaternionToGlobal([rigid_body.get(param) for param in rb_params])
        for i in range(4):
            rigid_body[rb_params[i]] = new_params[i]

        # Convert quaternions to Euler angles
        euler_angles = _quaternionToEuler([rigid_body.get(param) for param in rb_params])
        for i in range(3):
            rigid_body[rb_angles[i]] = euler_angles[i]


def _positionToGlobal(coords):
    R_motive_to_g = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])

    return np.matmul(R_motive_to_g, np.array(coords)).tolist()


def _quaternionToGlobal(args):
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

# def _extractObjects(markers, rigid_bodies):
#     global current_agents_id, current_manipulandums_id
#     rbs_id = list(rigid_bodies.keys())

#     agents_id, manipulandums_id = set(), set()
#     for rb_id in rbs_id:
#         (agents_id, manipulandums_id)[rb_id > 10].add(rb_id)

#     if len(markers) != 9 * len(agents_id) + 3 * len(manipulandums_id):
#         raise Exception('Wrong number of markers!')
    
#     agents_to_remove = current_agents_id.difference(agents_id)
#     for key in agents_to_remove:
#         del agents[key]

#     current_agents_id = agents_id

    manipulandums_to_remove = current_manipulandums_id.difference(manipulandums_id)
    for key in manipulandums_to_remove:
        del manipulandums[key]

    current_manipulandums_id = manipulandums_id

    for agent_id in current_agents_id:
        if agent_id in agents:
            agent = agents[agent_id]
        else:
            agent = agent.Agent(agent_id) 

        agent_head = rigid_bodies[agent_id]

        # Define the frame of the head LU
        agent.head.theta = _calcLUOrientation([marker for marker in markers.values() if marker['model_id'] == agent_id])
        agent.head.x = agent_head['x'] + LU_head_center_r*np.cos(agent.head.theta + LU_head_center_angle)
        agent.head.y = agent_head['y'] + LU_head_center_r*np.sin(agent.head.theta + LU_head_center_angle)

        agent.vsf.points, tail_marker = _rankPoints(markers, agent.head)

        last_vsf_points = [point.position for point in agent.vsf.points[3:]]
        alpha1 = _getAngle(last_vsf_points[0], last_vsf_points[1])
        alpha2 = _getAngle(last_vsf_points[1], tail_marker['pos'][:-1])

        # Define the frame of the tail LU
        agent.tail.position2d = [tail_marker['marker_x'], tail_marker['marker_y']]
        agent.tail.theta = 2 * alpha2 - alpha1

        # Define the body frame
        agent.position = [(agent.head.x + agent.tail.x) / 2, (agent.head.y + agent.tail.y) / 2]
        agent.theta = _getAngle(agent.head.position2d, agent.tail.position2d)

        agents[agent_id] = agent
    
    for key in agents:
        print(str(key) + ': ' + str(agents[key]))

    return agents, manipulandums


# def _rankPoints(markers, head_LU):

#     head_LU_pos = np.array(head_LU.position3d)

#     # Define remaining points and their id's that correspond to thei original indicies
#     remaining_points = [marker for marker in markers.values() if marker['model_id'] == 0]
#     for remaining_point in remaining_points:
#         remaining_point['pos'] = np.array([remaining_point.get(coord) for coord in m_pos])

#     # List of points ranks
#     rank = 1  # Current rank

#     # Find the point closest to the head LU
#     distances = np.linalg.norm([point['pos'] for point in remaining_points] - head_LU_pos, axis=1)

#     current_index = distances.argmin()

#     vsf_points = []
#     tail_marker = None

#     # Find the rank of the rest of the points
#     while remaining_points:
#         current_point = remaining_points.pop(current_index)

#         if rank != 6:
#             vsf_points.append(agent.VSFPoint(int(current_point['marker_id']), current_point['marker_x'], current_point['marker_y']))
#         else:
#             tail_marker = current_point

#         markers.get(str(current_point['model_id']) + '.' + str(current_point['marker_id']))['rank'] = rank

#         rank += 1  # Update rank

#         if(remaining_points):
#             distances = np.linalg.norm(
#                 [point['pos'] for point in remaining_points] - current_point['pos'], axis=1)
#             current_index = distances.argmin()

#     for marker in markers.values():
#         if marker['model_id'] != 0:
#             marker['rank'] = 0

#     return vsf_points, tail_marker

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
        markers.get(str(current_point['model_id']) + '.' + str(current_point['marker_id']))['rank'] = rank

        rank += 1  # Update rank

        if(remaining_points):
            distances = np.linalg.norm(
                [point['pos'] for point in remaining_points] - current_point['pos'], axis=1)
            current_index = distances.argmin()

    for marker in markers.values():
        if marker['model_id'] != 0:
            marker['rank'] = 0


def _calcLUOrientation(LU_head_markers):
    points = []

    for marker in LU_head_markers:
        points.append([marker['marker_x'], marker['marker_y']])

    points = np.array(points)
    triangle_sides = np.linalg.norm(points - np.roll(points, -1, 0), axis=1)

    i = triangle_sides.argsort()[1]
    i_next = i + 1 if i < 2 else 0

    delta = points[i, :] - points[i_next, :]
    theta = np.arctan2(delta[1], delta[0])

    return theta

def _calcManipOrientation(manipulandum_marker):
    points = []

    for marker in manipulandum_marker:
        points.append([marker['marker_x'], marker['marker_y']])

    points = np.array(points)
    triangle_sides = np.linalg.norm(points - np.roll(points, -1, 0), axis=1)

    i = triangle_sides.argsort()[0]
    i_next = i + 1 if i < 2 else 0

    delta = points[i, :] - points[i_next, :]
    theta = np.arctan2(delta[1], delta[0])

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

def _wheelsToBodyFrame(body_frame, LU_head_theta, LU_tail_theta, w):
    wheels_bf = w.copy()

    R_ob = np.array([[np.cos(body_frame[2]), -np.sin(body_frame[2])],
                     [np.sin(body_frame[2]), np.cos(body_frame[2])]])
    
    T_ob = np.block([[R_ob, np.array([body_frame[:2]]).T], [np.zeros((1,2)), 1]])
    T_bo = np.linalg.inv(T_ob)

    for i in range(4):
        w_b0 = [wheels_bf[i][0], wheels_bf[i][1], 1]
        wheels_bf[i] = np.matmul(T_bo, w_b0).T[:-1]
    for i in range(2):
        wheels_bf[i] = np.append(wheels_bf[i], (LU_head_theta - body_frame[2]) % (2 * np.pi) + global_var.BETA[i])

    for i in range(2, 4):
        wheels_bf[i] = np.append(wheels_bf[i], (LU_tail_theta - body_frame[2]) % (2 * np.pi) + global_var.BETA[i])

    return wheels_bf


# Calculate robot configuration from the mocap data
def getCurrentConfig(data):

    # Retreive markers and rigid_bodies data
    if len(data) == 1:
        markers, rigid_bodies = _unpackData(data[0])
    else:
        markers, rigid_bodies = copy.deepcopy(data)

    if len(markers) == 0 or len(rigid_bodies) == 0:
        raise Exception("No data received from Motive!")
    else:

        _convertData(markers, rigid_bodies)
        # _extractObjects(markers, rigid_bodies)       
                        
        # Get position of the head LU
        LU_head_rb = rigid_bodies[1]
        # Rank markers along the bridge
        _rankPoints(markers, LU_head_rb)
        
        # Define the frame of the head LU
        LU_head_theta = _calcLUOrientation([marker for marker in markers.values() if marker['model_id'] == 1])
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
        LU_tail_theta = 2 * alpha2 - alpha1

        # Define the frame of the tail LU
        LU_tail_frame = [LU_tail['marker_x'], LU_tail['marker_y'], LU_tail_theta]

        # Define the body frame
        body_frame = [(LU_head_x + LU_tail_frame[0]) / 2, (LU_head_y + LU_tail_frame[1]) / 2]
        body_frame_theta = _getAngle(LU_head_frame[:2],LU_tail_frame[:2])

        body_frame.append(body_frame_theta)
        
        # Define the frame of a manipulandum
        # manipulandum_rb = rigid_bodies[2]
        # manipulandum_theta = _calcManipOrientation([marker for marker in markers.values() if marker['model_id'] == 2])
        # manipulandum_frame = [manipulandum_rb['x'], manipulandum_rb['y'], manipulandum_theta]

        # Combine all frames
        # all_frames = [LU_head_frame, LU_tail_frame, body_frame, manipulandum_frame]
        all_frames = [LU_head_frame, LU_tail_frame, body_frame, [0, 0, 0]]

        # Calculate the wheels' coordinates from LU's coordinates
        wheels_global = __calcWheelsCoords(LU_head_frame, LU_tail_frame)
        wheels_bf = _wheelsToBodyFrame(body_frame, LU_head_theta, LU_tail_theta, wheels_global)

    return markers, all_frames, wheels_global, wheels_bf
