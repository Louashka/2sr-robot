import sys
from Motive import nat_net_client as nnc, mo_cap_data
import mas_controller
from Model import agent_old, global_var, manipulandum
import numpy as np
import pandas as pd
import Motive.motive_client as motive
import mas_controller
from typing import List

m_pos = ['marker_x', 'marker_y', 'marker_z']
pos_2d = ['marker_x', 'marker_y']
rb_pos = ['x', 'y', 'z']
rb_params = ['a', 'b', 'c', 'd']
rb_angles = ['roll', 'pitch', 'yaw']

class MocapReader:
    def __init__(self) -> None:
        self.__isRunning = False
        self.__data = None
        self.__mas = mas_controller.Swarm()
        self.__manipulandums = []

    @property
    def isRunning(self) -> bool:
        return self.__isRunning
    
    @isRunning.setter
    def isRunning(self, value: bool) -> None:
        self.__isRunning = value

    @property
    def mas(self) -> mas_controller.Swarm:
        return self.__mas
    
    @mas.setter
    def mas(self, value: mas_controller.Swarm) -> None:
        if not isinstance(value, mas_controller.Swarm):
            self.__mas = value

    @property
    def manipulandums(self) -> List[manipulandum.Shape]:
        return self.__manipulandums
    
    @manipulandums.setter
    def manipulandums(self, value: List[manipulandum.Shape]) -> None:
        for val in value:
            if not isinstance(val, manipulandum.Shape):
                raise ValueError('Wrong type of manipulandums!')
            
        self.__manipulandums = value

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
    
    def getCurrentConfig(self) -> tuple[dict, mas_controller.Swarm, manipulandum.Shape]:
        # markers, rigid_bodies = self.__unpackData()
        markers, rigid_bodies = self.__simulateData()

        self.__convertData(markers, rigid_bodies)

        if len(rigid_bodies) == 0:
            for robot in self.mas.agents:
                robot.status = False

            self.manipulandums = {}

        else:
            id_array = []
            for rb in rigid_bodies.values():

                id_array.append(rb['id'])

                if rb['id'] < 10:

                    head_markers = [marker for marker in markers.values() if marker['model_id'] == rb['id']]
                    head_pose = self.__calcHeadPose(rb, head_markers)

                    robot = self.mas.getAgentById(rb['id'])

                    if robot is None:
                        # Create a new agent
                        ranked_markers = self.__rankPoints(markers, head_pose[:-1])
                        if len(ranked_markers) != 6:
                            continue

                        vsf = agent_old.VSF(rb['id'], ranked_markers[:-1])

                        alpha1 = self.__getAngle(ranked_markers[2].position, ranked_markers[3].position)
                        alpha2 = self.__getAngle(ranked_markers[3].position, ranked_markers[4].position)
                        tail_theta = 2 * alpha2 - alpha1

                        tail_pose = [ranked_markers[-1].x, ranked_markers[-1].y, tail_theta]

                        robot_x = (head_pose[0] + tail_pose[0]) / 2
                        robot_y = (head_pose[1] + tail_pose[1]) / 2
                        robot_theta = self.__getAngle(head_pose[:-1], tail_pose[:-1])
                        robot_pose = [robot_x, robot_y, robot_theta]

                        head_wheels_pose = self.__calcWheelsCoords(robot_pose, head_pose, 'head')
                        tail_wheels_pose = self.__calcWheelsCoords(robot_pose, tail_pose, 'tail')

                        head_wheels = []
                        tail_wheels = []

                        for i in range(2):
                            head_wheels.append(agent_old.Wheel(rb['id'], i+1, head_wheels_pose[i]))
                            tail_wheels.append(agent_old.Wheel(rb['id'], i+3, tail_wheels_pose[i]))

                        head = agent_old.LU(rb['id'], head_pose, head_wheels)
                        tail = agent_old.LU(rb['id'], tail_pose, tail_wheels, ranked_markers[-1].marker_id)

                        robot = agent_old.Robot(rb['id'], robot_pose, head, tail, vsf)
                        self.mas.agents.append(robot)
                    else:
                        # Update existing agent
                        robot.head.pose = head_pose

                        for vsf_marker in robot.vsf.markers:
                            updated_vsf_marker = markers['0.' + str(vsf_marker.marker_id)]
                            vsf_marker.x = updated_vsf_marker['marker_x']
                            vsf_marker.y = updated_vsf_marker['marker_y']

                        alpha1 = self.__getAngle(robot.vsf.markers[2].position, robot.vsf.markers[3].position)
                        alpha2 = self.__getAngle(robot.vsf.markers[3].position, robot.vsf.markers[4].position)
                        tail_theta = 2 * alpha2 - alpha1

                        tail_marker = markers['0.' + str(robot.tail.marker_id)]
                        robot.tail.pose = [tail_marker['marker_x'], tail_marker['marker_y'], tail_theta]

                        robot.x = (robot.head.x + robot.tail.y) / 2
                        robot.y = (robot.head.y + robot.tail.y) / 2
                        robot.theta = self.__getAngle(robot.head.position, robot.tail.position)

                        head_wheels = self.__calcWheelsCoords(robot.pose, robot.head.pose, 'head')
                        tail_wheels = self.__calcWheelsCoords(robot.pose, robot.tail.pose, 'tail')
                        for i in range(2):
                            robot.tail.wheels[i].pose = head_wheels[i]

                        robot.status = True
                        
                    # Disable the agent if communication with it is lost
                    for robot in self.mas.agents:
                        if robot.id not in id_array:
                            robot.status = False
                else:
                    # Unpack a manipulandum
                    pass


        return [markers, self.mas, self.manipulandums]
    
    def __unpackData(self) -> list:
        if self.__data is None:
            raise Exception('No data received from Motive!')
        
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
    
    def __simulateData(self):
        pose = 2

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
    
    def __calcHeadPose(self, rb, markers):
        points = []

        for marker in markers:
            points.append([marker['marker_x'], marker['marker_y']])

        points = np.array(points)
        triangle_sides = np.linalg.norm(points - np.roll(points, -1, 0), axis=1)

        i = triangle_sides.argsort()[1]
        i_next = i + 1 if i < 2 else 0

        delta = points[i, :] - points[i_next, :]
        theta = np.arctan2(delta[1], delta[0]) + np.pi
        
        x = rb['x'] + global_var.HEAD_CENTER_R * np.cos(theta + global_var.HEAD_CENTER_ANGLE)
        y = rb['y'] + global_var.HEAD_CENTER_R * np.sin(theta + global_var.HEAD_CENTER_ANGLE)

        return [x, y, theta]
    
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
        while len(ranked_markers) < 6:
            current_point = remaining_points.pop(current_index)
            vsf_marker = agent_old.Marker(current_point['marker_id'], current_point['marker_x'], current_point['marker_y'])
            ranked_markers.append(vsf_marker)

            if(remaining_points):
                distances = np.linalg.norm(
                    [point['pos'] for point in remaining_points] - current_point['pos'], axis=1)
                current_index = distances.argmin()

        return ranked_markers




