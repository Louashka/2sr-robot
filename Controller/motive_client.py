import sys
from Motive import nat_net_client as nnc, mo_cap_data
from Model import agent_old, global_var, manipulandum, robot2sr
import numpy as np
import pandas as pd
from typing import List
from scipy.interpolate import splprep, splev
from circle_fit import taubinSVD

m_pos = ['marker_x', 'marker_y', 'marker_z']
pos_2d = ['marker_x', 'marker_y']
rb_pos = ['x', 'y', 'z']
rb_params = ['a', 'b', 'c', 'd']
rb_angles = ['roll', 'pitch', 'yaw']

class MocapReader:
    def __init__(self) -> None:
        self.__isRunning = False
        self.__data = None

    @property
    def isRunning(self) -> bool:
        return self.__isRunning
    
    @isRunning.setter
    def isRunning(self, value: bool) -> None:
        self.__isRunning = value

    @property
    def data(self) -> mo_cap_data.MoCapData:
        return self.__data

    def __receiveData(self, value) -> None:
        self.__data = value

    def __parseArgs(self, arg_list, args_dict) -> dict:
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
    
    def startDataListener(self) -> None:
        options_dict = {}
        options_dict["clientAddress"] = "127.0.0.1"
        options_dict["serverAddress"] = "127.0.0.1"
        options_dict["use_multicast"] = True

        # Start Motive streaning
        options_dict = self.__parseArgs(sys.argv, options_dict)

        streaming_client = nnc.NatNetClient()
        streaming_client.set_client_address(options_dict["clientAddress"])
        streaming_client.set_server_address(options_dict["serverAddress"])
        streaming_client.set_use_multicast(options_dict["use_multicast"])

        streaming_client.mocap_data_listener = self.__receiveData

        self.isRunning = streaming_client.run()

    def getAgentConfig(self) -> tuple[dict, dict, List[agent_old.Marker]]:
        agent = {}

        markers, rigid_bodies = self.__unpackData()

        if markers is None or rigid_bodies is None:
            return {}, {}, []
        
        if not markers or not rigid_bodies:
            return {}, {}, []

        self.__convertData(markers, rigid_bodies)

        if len(rigid_bodies) == 1 and len(markers) == 9:
            rb = list(rigid_bodies.values())[0]

            agent['id'] = rb['id']

            # Calculate the pose of the head LU
            head_markers = [marker for marker in markers.values() if marker['model_id'] == rb['id']]
            head_pose, start_marker = self.__calcHeadPose(rb, head_markers)
            agent['head'] = head_pose

            # Sort markers to within the VSF
            ranked_markers = self.__rankPoints(markers, head_pose[:-1])
            # If some markers are missing we ignore the robot
            if len(ranked_markers) != 6:
                return {}, {}, []
            
            # Calculate the pose of the tail LU
            # alpha1 = self.__getAngle(ranked_markers[2].position, ranked_markers[3].position)
            # alpha2 = self.__getAngle(ranked_markers[3].position, ranked_markers[4].position)
            # tail_theta = 2 * alpha2 - alpha1
            tail_theta = self.__getAngle(ranked_markers[-2].position, ranked_markers[-1].position) + 0.31615

            tail_pose = [ranked_markers[-1].x, ranked_markers[-1].y, tail_theta]
            agent['tail'] = tail_pose
            
            # Calculate the robot's position
            robot_x, robot_y, robot_theta = self.__findMidpoint(ranked_markers[:-1])

            # Calculate VSS' curvatures
            k1 = self.__calculate_curvature(ranked_markers[0].position, [robot_x, robot_y], head_pose[-1], robot_theta)
            k2 = self.__calculate_curvature([robot_x, robot_y], ranked_markers[4].position, robot_theta, tail_theta)
            # k1 = (robot_theta - head_pose[-1]) / global_var.L_VSS
            # k2 = (tail_theta - robot_theta) / global_var.L_VSS
            agent['k1'] = k1
            agent['k2'] = k2

            # Calculate the robot's orientation
            # robot_theta = self.__getAngle(head_pose[:-1], tail_pose[:-1])

            # theta1 = head_pose[-1] + k1 * global_var.L_VSS
            # theta2 = tail_pose[-1] - k2 * global_var.L_VSS

            # theta1 = self.adjustOrientation(head_pose[:-1], tail_pose[:-1], [robot_x, robot_y], theta1)
            # theta2 = self.adjustOrientation(head_pose[:-1], tail_pose[:-1], [robot_x, robot_y], theta2)

            # robot_theta = (theta1 + theta2) / 2
            
            agent['x'] = robot_x
            agent['y'] = robot_y
            agent['theta'] = robot_theta

            # print(agent)

        else:
            return {}, {}, []

        return agent, markers, ranked_markers
    
    def __findMidpoint(self, points:List[agent_old.Marker]):
        x = []
        y = []

        for point in points:
            x.append(point.x)
            y.append(point.y)
            
        # Fit a spline curve to the data points
        tck, _ = splprep([x, y], s=0)

        # Calculate the arc length of the spline curve
        s = np.linspace(0, 1, 1000)
        spline_x, spline_y = splev(s, tck)
        spline_arc_length = 0
        for i in range(1, len(s)):
            spline_arc_length += np.sqrt((spline_x[i] - spline_x[i-1])**2 + (spline_y[i] - spline_y[i-1])**2)

        # Find the midpoint of the spline curve
        spline_arc_length_midpoint = spline_arc_length / 2
        current_arc_length = 0
        for i in range(1, len(s)):
            current_arc_length += np.sqrt((spline_x[i] - spline_x[i-1])**2 + (spline_y[i] - spline_y[i-1])**2)
            if current_arc_length >= spline_arc_length_midpoint:
                midpoint_s = s[i-1]
                midpoint_x, midpoint_y = splev(midpoint_s, tck)
                break

        # Calculate the tangent vector at the midpoint
        tangent_x, tangent_y = splev(midpoint_s, tck, der=1)
        theta = np.arctan2(tangent_y, tangent_x)
    
        # If the curve has only one point, return that point
        return midpoint_x, midpoint_y, theta
    
    def adjustOrientation(self, A, B, C, theta):
        vector_AC = (C[0] - A[0], C[1] - A[1])  # Vector from A to C
        vector_BC = (C[0] - B[0], C[1] - B[1])  # Vector from B to C

        cross_product = vector_AC[0] * vector_BC[1] - vector_AC[1] * vector_BC[0]

        if cross_product < 0:
            theta += np.pi

        theta %= (2 * np.pi)

        return theta
    
    # def getCurrentConfig(self) -> tuple[List[dict], List[dict]]:
    #     agents = []
    #     manipulandums = []

    #     # markers, rigid_bodies = self.__unpackData()
    #     markers, rigid_bodies = self.__simulateData()

    #     self.__convertData(markers, rigid_bodies)

    #     if len(rigid_bodies) != 0:
    #         id_array = []
    #         for rb in rigid_bodies.values():

    #             id_array.append(rb['id'])

    #             # Agent 'head' locomotion units have id's < 10, manipulandums have id's > 10 
    #             if rb['id'] < 10:

    #                 robot = {} # Create a dict that stores robot's data
    #                 robot['id'] = rb['id']

    #                 # Calculate the pose of the head LU
    #                 head_markers = [marker for marker in markers.values() if marker['model_id'] == rb['id']]
    #                 head_pose = self.__calcHeadPose(rb, head_markers)

    #                 # Sort markers to within the VSF
    #                 ranked_markers = self.__rankPoints(markers, head_pose[:-1])
    #                 # If some markers are missing we ignore the robot
    #                 if len(ranked_markers) != 6:
    #                     continue

    #                 # Calculate VSS' curvatures
    #                 k1 = self._calculate_curvature(ranked_markers[:3], global_var.L_VSS)
    #                 k2 = self._calculate_curvature(ranked_markers[2:], global_var.L_VSS)
    #                 robot['k'] = [k1, k2]

    #                 # Calculate the pose of the tail LU
    #                 alpha1 = self.__getAngle(ranked_markers[2].position, ranked_markers[3].position)
    #                 alpha2 = self.__getAngle(ranked_markers[3].position, ranked_markers[4].position)
    #                 tail_theta = 2 * alpha2 - alpha1

    #                 tail_pose = [ranked_markers[-1].x, ranked_markers[-1].y, tail_theta]

    #                 # Calculate the robot pose
    #                 robot_x = (head_pose[0] + tail_pose[0]) / 2
    #                 robot_y = (head_pose[1] + tail_pose[1]) / 2
    #                 robot_theta = self.__getAngle(head_pose[:-1], tail_pose[:-1])
                    
    #                 robot['x'] = robot_x
    #                 robot['y'] = robot_y
    #                 robot['theta'] = robot_theta

    #                 print(robot)

    #                 agents.append(robot)
    #             else:
    #                 # Unpack a manipulandum
    #                 pass


    #     return [agents, manipulandums]
    
    def __unpackData(self) -> tuple[dict, dict]:
        if self.data is None:
            # raise ValueError('No data received from Motive!')
            return None, None
        
        rigid_body_data = self.data.rigid_body_data
        labeled_marker_data = self.data.labeled_marker_data

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
    
    def __simulateData(self) -> tuple[dict, dict]:
        pose = 3

        markers_df = pd.read_csv('Data/markers.csv')
        rigid_bodies_df = pd.read_csv('Data/rigid_bodies.csv')

        markers_df_ = markers_df[markers_df["pose"]
                                    == pose].drop('pose', axis=1)
        rigid_bodies_df_ = rigid_bodies_df[rigid_bodies_df["pose"] == pose].drop(
            'pose', axis=1)

        markers = {}
        rigid_bodies = {}

        for index, row in markers_df_.iterrows():
            marker = row.to_dict()
            marker['model_id'] = int(marker['model_id'])
            marker['marker_id'] = int(marker['marker_id'])
            markers[str(marker['model_id']) + '.' + str(marker['marker_id'])] = marker

        for index, row in rigid_bodies_df_.iterrows():
            rigid_body = row.to_dict()
            rigid_body['id'] = int(rigid_body['id'])
            rigid_bodies[rigid_body['id']] = rigid_body        

        return markers, rigid_bodies

    def __convertData(self, markers: dict, rigid_bodies: dict):
        # Convert values from the Motive frame to the global frame
        for marker in markers.values():
            new_pos = self.__positionToGlobal([marker.get(coord) for coord in m_pos])
            for i in range(3):
                marker[m_pos[i]] = new_pos[i]

        for rigid_body in rigid_bodies.values():
            new_pos = self.__positionToGlobal([rigid_body.get(coord) for coord in rb_pos])
            for i in range(3):
                rigid_body[rb_pos[i]] = new_pos[i]

            new_params = self.__quaternionToGlobal([rigid_body.get(param) for param in rb_params])
            for i in range(4):
                rigid_body[rb_params[i]] = new_params[i]

            # Convert quaternions to Euler angles
            euler_angles = self.__quaternionToEuler([rigid_body.get(param) for param in rb_params])
            for i in range(3):
                rigid_body[rb_angles[i]] = euler_angles[i]

    def __positionToGlobal(self, coords: list):
        R_motive_to_g = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])

        return np.matmul(R_motive_to_g, np.array(coords)).tolist()


    def __quaternionToGlobal(self, args):
        return [-args[0], args[2], args[1], args[3]]


    def __quaternionToEuler(self, coeffs):
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
    
    def __calcHeadPose(self, rb, markers) -> tuple[list, agent_old.Marker]:
        points = []

        for marker in markers:
            points.append([marker['marker_x'], marker['marker_y']])

        points = np.array(points)
        triangle_sides = np.linalg.norm(points - np.roll(points, -1, 0), axis=1)

        i = triangle_sides.argsort()[1]
        i_next = i + 1 if i < 2 else 0

        delta = points[i_next, :] - points[i, :]
        # theta = np.arctan2(delta[1], delta[0]) + np.pi
        theta = np.arctan2(delta[1], delta[0])
        
        x = rb['x'] + global_var.HEAD_CENTER_R * np.cos(theta + global_var.HEAD_CENTER_ANGLE)
        y = rb['y'] + global_var.HEAD_CENTER_R * np.sin(theta + global_var.HEAD_CENTER_ANGLE)

        start_marker = agent_old.Marker(markers[i]['marker_id'], markers[i]['marker_x'], markers[i]['marker_y'])

        return [x, y, theta], start_marker
    
    def __getAngle(self, p1: list, p2: list):
        a = p2[0] - p1[0]
        b = p2[1] - p1[1]

        alpha = np.arctan2(b, a)

        return alpha
    
    def __calcWheelsCoords(self, robot_pose, LU_pose, LU_type):
        if LU_type == 'head':
            w1_0 = np.array([[-0.0275], [0]])
            w2_0 = np.array([[0.0105], [-0.0275]])
        elif LU_type == 'tail':
            w1_0 = np.array([[0.0275], [0]])
            w2_0 = np.array([[-0.0105], [-0.027]])

        R = np.array([[np.cos(LU_pose[2]), -np.sin(LU_pose[2])],
                    [np.sin(LU_pose[2]), np.cos(LU_pose[2])]])
        w1 = np.matmul(R, w1_0).T[0] + LU_pose[:2]
        w2 = np.matmul(R, w2_0).T[0] + LU_pose[:2]

        w = self.__wheelsToBodyFrame(robot_pose, LU_pose[-1], [w1, w2], LU_type)

        return w

    def __wheelsToBodyFrame(self, robot_pose, LU_theta, w, LU_type):
        wheels = w.copy()

        R_ob = np.array([[np.cos(robot_pose[2]), -np.sin(robot_pose[2])],
                        [np.sin(robot_pose[2]), np.cos(robot_pose[2])]])
        
        T_ob = np.block([[R_ob, np.array([robot_pose[:2]]).T], [np.zeros((1,2)), 1]])
        T_bo = np.linalg.inv(T_ob)

        if LU_type == 'head':
            offset = 0
        elif LU_type == 'tail':
            offset = 2

        for i in range(2):
            w_b0 = [wheels[i][0], wheels[i][1], 1]
            wheels[i] = np.matmul(T_bo, w_b0).T[:-1]
            # wheels[i] = [wheels[i][0], wheels[i][1]]
            wheels[i] = np.append(wheels[i], (LU_theta - robot_pose[2]) % (2 * np.pi) + global_var.BETA[i+offset])

        return wheels
    
    def __rankPoints(self, markers, head_position) -> List[agent_old.Marker]:
        head_position = np.array(head_position)

        # Define remaining points and their id's that correspond to thei original indicies
        remaining_points = [marker for marker in markers.values() if marker['model_id'] == 0]
        for remaining_point in remaining_points:
            remaining_point['pos'] = np.array([remaining_point.get(coord) for coord in pos_2d])

        # List of ranked points
        ranked_markers = []

        # Find the point closest to the head LU
        distances = np.linalg.norm([point['pos'] for point in remaining_points] - head_position, axis=1)

        current_index = distances.argmin()

        # Find the rank of the rest of the points
        while len(ranked_markers) < 6 and remaining_points:
            current_point = remaining_points.pop(current_index)
            vsf_marker = agent_old.Marker(current_point['marker_id'], current_point['marker_x'], current_point['marker_y'])
            ranked_markers.append(vsf_marker)

            if(remaining_points):
                distances = np.linalg.norm(
                    [point['pos'] for point in remaining_points] - current_point['pos'], axis=1)
                current_index = distances.argmin()

        return ranked_markers
    
    def __calculate_curvature(self, p0, p, theta0, theta):
        # for the segment_points, the input should be in the following format:
        # [end1, center, end2]
        # point A, point B, point C

        # segment_length is in meter, and the value is 0.077m

        # Calculate the distances between the points
        # print(segment_points[0])
        # a = np.sqrt(
        #     (segment_points[0].x - segment_points[1].x) ** 2 + (segment_points[0].y - segment_points[1].y) ** 2)
        # b = np.sqrt(
        #     (segment_points[2].x - segment_points[1].x) ** 2 + (segment_points[2].y - segment_points[1].y) ** 2)
        # c = np.sqrt(
        #     (segment_points[0].x - segment_points[2].x) ** 2 + (segment_points[0].y - segment_points[2].y) ** 2)

        # # Calculate the radius
        # alpha = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))
        # # print("alpha", alpha)
        # r = a / (2 * np.sin(alpha))
        # # print("radius", r)

        # # Calculate the central angle
        # theta = segment_length / r
        # # print("central angle", theta)

        # # Calculate the curvature, should be positive value all time (need to be comfirn)
        # curvature = theta / segment_length

        # Calculate the angle subtended by the arc
        # v1 = (segment_points[1].x - segment_points[0].x, segment_points[1].y - segment_points[0].y)
        # v2 = (segment_points[2].x - segment_points[1].x, segment_points[2].y - segment_points[1].y)
        # dot_product = v1[0]*v2[0] + v1[1]*v2[1]
        # cross_product = v1[0]*v2[1] - v1[1]*v2[0]
        # theta = np.arctan2(cross_product, dot_product)
        
        # # Calculate the curvature
        # curvature = theta / segment_length

        # Determine the sign by slope
        # Normalize the segment and compute the reletive angle to the end1
        # theta1 = np.arctan2(segment_points[1].x - segment_points[0].x, segment_points[1].y - segment_points[0].y)
        # theta2 = np.arctan2(segment_points[2].x - segment_points[0].x, segment_points[2].y - segment_points[0].y)

        # print("theta1: ", theta1)
        # print("theta2: ", theta2)

        # if v1 > v2:
        #     curvature = - curvature

        # points = []

        # for point in segment_points:
        #     points.append(point.position)

        # xc, yc, r, sigma = taubinSVD(points)
        # curvature = 1 / r

        # if seg == 1:
        #     start_point = points[-1]
        # else:
        #     start_point = points[0]

        # d = np.sqrt((start_point[0] - xc)**2 + (start_point[1] - yc)**2)
        # x = xc + (r**2 / d) * (start_point[0] - xc)
        # y = yc + (r**2 / d) * (start_point[1] - yc)

        # theta = np.arctan2(-(x - xc),(y - yc))

        # w = np.sqrt((segment_points[0].x - segment_points[1].x) ** 2 + (segment_points[0].y - segment_points[1].y) ** 2)
        # mid_point = [(segment_points[-1].x - segment_points[0].x) / 2, (segment_points[-1].y - segment_points[0].y) / 2]
        # h = np.sqrt((mid_point[0] - xc)**2 + (mid_point[1] - yc)**2)

        # r = w**2 / (8*h) + h/2
        # curvature = 1 / r

        # # Calculate the tangent vector
        # tangent_vector = np.array([points[-1][0] - points[0][0], points[-1][1] - points[0][1]])

        # # Calculate the vector from the center of curvature to the current point
        # center_vector = np.array([points[0][0] - xc, points[0][1] - yc])

        # # Calculate the sign of the curvature based on the tangent direction
        # tangent_direction = np.cross(tangent_vector, center_vector)
        # sign = 1 if tangent_direction <= 0 else -1

        # # Apply the sign to the curvature
        # signed_curvature = sign * curvature

        curvature_x = (np.sin(theta) - np.sin(theta0)) / (p[0] - p0[0])
        curvature_y = (-np.cos(theta) + np.cos(theta0)) / (p[1] - p0[1])

        curvature = (curvature_x + curvature_y) / 2

        return curvature




