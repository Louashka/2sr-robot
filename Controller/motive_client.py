import sys
from Motive import nat_net_client as nnc, mo_cap_data
from Model import global_var
import numpy as np
import pandas as pd
from typing import List
from scipy.optimize import minimize 
from scipy.interpolate import splprep, splev
from circle_fit import taubinSVD

m_pos = ['marker_x', 'marker_y', 'marker_z']
pos_2d = ['marker_x', 'marker_y']
rb_pos = ['x', 'y', 'z']
rb_params = ['a', 'b', 'c', 'd']
rb_angles = ['roll', 'pitch', 'yaw']

class Marker():
    def __init__(self, marker_id: int, x: float, y: float) -> None:
        self.__marker_id = marker_id
        self.x = x
        self.y = y

    @property
    def marker_id(self) -> int:
        return self.__marker_id

    @property
    def x(self) -> float:
        return self.__x
    
    @x.setter
    def x(self, value: float) -> None:
        self.__x = value

    @property
    def y(self) -> float:
        return self.__y
    
    @y.setter
    def y(self, value: float) -> None:
        self.__y = value

    @property
    def position(self) -> list:
        return [self.x, self.y]

class MocapReader:
    def __init__(self) -> None:
        self.__isRunning = False
        self.__data = None
        self.__markers_id = set()
        self.agent_theta_prev = None

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

    """
    Determine the current configuration of the single 2SR agent
    :return: dict, agent's generalized coordinates; dict, markers' id and coordinates
    """

    def getAgentConfig(self, agents_exp_n = 1, manip_exp_n = 0) -> tuple[dict, dict, dict, str]:
        agent = {}
        manip = {}

        # Parse Motive data
        markers, rigid_bodies = self.__unpackData()
        # markers, rigid_bodies = self.__simulateData()

        if markers is None or rigid_bodies is None:
            return {}, {}, {}, 'No data is received from Motive!'
        
        if not markers or not rigid_bodies:
            return {}, {}, {}, 'No markers or rigid bodies are detected!'

        # Convert values from the Motive frame to the global frame
        self.__convertData(markers, rigid_bodies)

        exp_n = agents_exp_n + manip_exp_n

        if len(rigid_bodies) == exp_n and len(markers) == 9 * agents_exp_n + 3 * manip_exp_n:
            rb_agent = rigid_bodies[1]

            agent['id'] = rb_agent['id']

            # Calculate the pose of the head LU
            head_markers = [marker for marker in markers.values() if marker['model_id'] == rb_agent['id']]
            if len(head_markers) != 3:
                return {}, {}, {}, 'Problem with the agent\'s model id!'
            
            head_pose = self.__calcRbPose(rb_agent, head_markers)
            agent['head'] = head_pose

            # Sort markers to within the VSF
            ranked_markers = self.__rankPoints(markers, head_pose[:-1])
            # If some markers are missing we ignore the robot
            if len(ranked_markers) != 6:
                return {}, {}, {}, 'Not enough markers in VSF!'
            
            if len(self.__markers_id) == 0:
                for marker in markers.values():
                    self.__markers_id.add(marker['marker_id'])
            
            # Calculate the pose of the tail LU
            tail_theta = self.__getAngle(ranked_markers[-2].position, ranked_markers[-1].position) + 0.31615
            # tail_theta = tail_theta % (2 * np.pi)

            tail_pose = [ranked_markers[-1].x, ranked_markers[-1].y, tail_theta]
            agent['tail'] = tail_pose
            
            # Calculate the robot's configuration
            robot_x, robot_y, robot_theta, k1, k2 = self.__extrapolateCurve(ranked_markers[:-1])

            agent['x'] = robot_x
            agent['y'] = robot_y
            agent['theta'] = robot_theta
            agent['k1'] = k1
            agent['k2'] = k2

            # print(agent)

            if manip_exp_n == 1:
                rb_manip = rigid_bodies[2]

                manip['id'] = rb_manip['id']

                manip_markers = [marker for marker in markers.values() if marker['model_id'] == rb_manip['id']]
                if len(manip_markers) != 3:
                    return {}, {}, {}, 'Problem with the manipulandum\'s model id!'
            
                manip_pose = self.__calcRbPose(rb_manip, manip_markers)

                manip['x'] = manip_pose[0]
                manip['y'] = manip_pose[1]
                manip['theta'] = manip_pose[2]
        else:
            return {}, {}, {}, 'Wrong number of the markers or rigid bodies!'

        return agent, manip, markers, 'Configuration successfully extracted.'
    
    
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
            if len(self.__markers_id) != 0 and marker_id not in self.__markers_id:
                continue
            marker = {'model_id': model_id, 'marker_id': marker_id,
                        'marker_x': -marker.pos[0], 'marker_y': marker.pos[2], 'marker_z': marker.pos[1]}
            markers[str(model_id) + '.' + str(marker_id)] = marker

        for rigid_body in rigid_body_list:
            rigid_body = {'id': int(rigid_body.id_num), 'x': -rigid_body.pos[0], 'y': rigid_body.pos[2], 'z': rigid_body.pos[1], 'a': rigid_body.rot[0],
                            'b': rigid_body.rot[1], 'c': rigid_body.rot[2], 'd': rigid_body.rot[3]}
            rigid_bodies[int(rigid_body['id'])] = rigid_body    

        return markers, rigid_bodies
    
    def __simulateData(self) -> tuple[dict, dict]:
        pose = 1

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
        # for marker in markers.values():
            # new_pos = self.__positionToGlobal([marker.get(coord) for coord in m_pos])
            # for i in range(3):
            #     marker[m_pos[i]] = new_pos[i]

        for rigid_body in rigid_bodies.values():
            # new_pos = self.__positionToGlobal([rigid_body.get(coord) for coord in rb_pos])
            # for i in range(3):
            #     rigid_body[rb_pos[i]] = new_pos[i]

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
    
    def __calcRbPose(self, rb, markers, tp='head') -> List[float]:
        points = []

        for marker in markers:
            points.append([marker['marker_x'], marker['marker_y']])

        points = np.array(points)
        triangle_sides = np.linalg.norm(points - np.roll(points, -1, 0), axis=1)

        i = triangle_sides.argsort()[1]
        i_next = i + 1 if i < 2 else 0

        if triangle_sides[i-1] > triangle_sides[i_next]:
            i, i_next = i_next, i

        delta = points[i_next, :] - points[i, :]
        theta = np.arctan2(delta[1], delta[0])
        # theta = theta  % (2 * np.pi)
        
        if tp == 'head':
            x = rb['x'] + global_var.HEAD_CENTER_R * np.cos(theta + global_var.HEAD_CENTER_ANGLE)
            y = rb['y'] + global_var.HEAD_CENTER_R * np.sin(theta + global_var.HEAD_CENTER_ANGLE)
        elif tp == 'manip':
            x = rb['x']
            y = rb['y']

        return [x, y, theta]
    
    def __rankPoints(self, markers, head_position) -> List[Marker]:
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
            vsf_marker = Marker(current_point['marker_id'], current_point['marker_x'], current_point['marker_y'])
            ranked_markers.append(vsf_marker)

            if(remaining_points):
                distances = np.linalg.norm(
                    [point['pos'] for point in remaining_points] - current_point['pos'], axis=1)
                current_index = distances.argmin()

        return ranked_markers
    
    def __getAngle(self, p1: list, p2: list):
        a = p2[0] - p1[0]
        b = p2[1] - p1[1]

        alpha = np.arctan2(b, a)

        return alpha
    
    def __extrapolateCurve(self, points:List[Marker]) -> tuple[float, float, float, float, float]:
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

        if self.agent_theta_prev is None:
            self.agent_theta_prev = theta
        else:
            diff = theta - self.agent_theta_prev
            # Check if we've crossed the -pi/pi boundary
            offset = 0
            if diff > np.pi:
                offset -= 2*np.pi
            elif diff < -np.pi:
                offset += 2*np.pi

            theta += offset
            self.agent_theta_prev = theta

        x_der1, y_der1 = splev(s[:500], tck, der=1)
        x_der2, y_der2 = splev(s[:500], tck, der=2)

        curvature = (x_der1 * y_der2 - y_der1 * x_der2) / np.power(x_der1** 2 + y_der1** 2, 3 / 2)
        k1 = curvature.mean()
        
        x_der1, y_der1 = splev(s[500:], tck, der=1)
        x_der2, y_der2 = splev(s[500:], tck, der=2)

        curvature = (x_der1 * y_der2 - y_der1 * x_der2) / np.power(x_der1** 2 + y_der1** 2, 3 / 2)
        k2 = curvature.mean()
    
        # If the curve has only one point, return that point
        return midpoint_x.item(), midpoint_y.item(), theta, k1, k2