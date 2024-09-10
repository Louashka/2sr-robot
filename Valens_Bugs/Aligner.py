import math
import json
import cv2
import cv2.aruco as aruco
import numpy as np
import matplotlib.pyplot as plot

import sys
import nat_net_client as nnc

'''
This is the code for aligning the coordinate of optitrack and camera
Besides, it will also load the trajectory in the camera frame and display it

The algorithm of this code is:
1. Read the OptiTrack markers' positions (unit: meter, coordinate respective to workspace: (-x, z, y))
2. Read the pixel position and the rotation vector of ArUcos
3. Align the center of OptiTrack marker and the ArUco, calculate the positions
    of ArUco in the OptiTrack's coordinate
4. Calculate the distance between two ArUco markers (meter and pixel) to calculate
    the ratio of 'pixel:meter', for converting the trajectory (in meter) to pixel coordinate
5. Read the target trajectory and convert it into pixels
6. Display the trajectory in the camera frames

Relate to the main tasks, multiple related function will be includes, such as undistortion and image cropping
'''
path_y = np.arange(0, 2, 0.01)
path_x = np.array([np.sin(y / 0.21) * y / 2.7 for y in path_y])
# shift
path_y = path_y - 1
path_x = path_x 
# print("path x: ", path_x)
# print("path y: ", path_y)

class Aligner():

    def __init__(self):
        self.file_path = "Valens_Bugs/calibration_data.json"
        self.mocap_data = None

        self.aruco_A_id = 10
        self.aruco_B_id = 12

        try:
            with open(self.file_path, "r") as json_file:
                data = json.load(json_file)
                self.data = data
        except FileNotFoundError:
            print(f"Error: {self.file_path} not found.")
            return

        self.read_camera_calibration_data()

    def calibration(self):
        self.startDataListener()

        cap = cv2.VideoCapture(0)
        cap.set(3, 1280)
        cap.set(4, 720)

        while cap.isOpened():

            if self.mocap_data is None:
                continue

            labeled_marker_data = self.mocap_data.labeled_marker_data
            labeled_marker_list = labeled_marker_data.labeled_marker_list

            if len(labeled_marker_list) != 4:
                continue

            markers_A = []
            markers_B = []

            for marker in labeled_marker_list:
                if marker.pos[2] < 0:
                    markers_A.append(self.convert_opti_coordinate_to_workspace(marker.pos))
                else:
                    markers_B.append(self.convert_opti_coordinate_to_workspace(marker.pos))

            opti_A_position = np.array([(markers_A[0][0] + markers_A[1][0]) / 2,
                                        (markers_A[0][1] + markers_A[1][1]) / 2])      
            opti_B_position = np.array([(markers_B[0][0] + markers_B[1][0]) / 2,
                                        (markers_B[0][1] + markers_B[1][1]) / 2])   

            print("optiTrack marker set A: ", opti_A_position)
            print("optiTrack marker set B: ", opti_B_position)


            _, frame = cap.read()
            undistort_frame = self.undistort(frame)
            undistort_frame = cv2.rotate(undistort_frame, cv2.ROTATE_180)
            # cv2.imshow("rotated by 180", rotated_frame)

            _, aruco_A_position, aruco_B_position = self.DetectArucoPose(undistort_frame)
            if aruco_A_position is None or aruco_B_position is None:
                continue
            
            aruco_pixel_distance_x = aruco_B_position[0] - aruco_A_position[0]
            aruco_pixel_distance_y = aruco_B_position[1] - aruco_A_position[1]

            # Calculate the Region of Interest (ROI) based on the Aruco marker positions
            roi_x = int(min(aruco_A_position[0], aruco_B_position[0]))
            roi_y = int(min(aruco_A_position[1], aruco_B_position[1]))
            roi_width = int(abs(aruco_pixel_distance_x))
            roi_height = int(abs(aruco_pixel_distance_y))

            # Crop the frame using the calculated ROI
            cropped_frame = undistort_frame[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width]
            # print(cropped_frame.shape)

            # Display the cropped frame
            # cv2.imshow('Cropped Frame', cropped_frame)

            # Resize the cropped frame while maintaining the aspect ratio
            height = 720
            width = int(cropped_frame.shape[1] * (height / cropped_frame.shape[0]))
            resized_frame = cv2.resize(cropped_frame, (width, height), interpolation=cv2.INTER_AREA)
            frame_size = np.array([resized_frame.shape[1], resized_frame.shape[0]])
            print("resize frame shape: ", frame_size)

            # Display the resized frame
            cv2.imshow('Resized Frame', resized_frame)

            # convert the opti coordinate back to camera coordinate
            scale_x = frame_size[0] / abs(opti_B_position[1] - opti_A_position[1])
            scale_y = frame_size[1] / abs(opti_B_position[0] - opti_A_position[0])

            translate_x = -scale_x * opti_A_position[1]
            translate_y = -scale_y * opti_A_position[0]

            transformation = [scale_x, scale_y, translate_x, translate_y]

            optiA_cam_pos = self.convert_opti_coordinate_to_camera_coordinate(opti_A_position, transformation)
            optiB_cam_pos = self.convert_opti_coordinate_to_camera_coordinate(opti_B_position, transformation)

            print("optiA_cam_pos: ", optiA_cam_pos)
            print("optiB_cam_pos: ", optiB_cam_pos)
            cv2.circle(resized_frame, (int(optiA_cam_pos[0]), int(optiA_cam_pos[1])), 30, (0, 0, 255), -1)
            cv2.circle(resized_frame, (int(optiB_cam_pos[0]), int(optiB_cam_pos[1])), 30, (0, 255, 0), -1)
            # cv2.imshow("result", resized_frame)

            # map the trajectory to the coordinate
            for i in range (0, len(path_x)):
                point = np.array([path_x[i], path_y[i]])
                camera_position = self.convert_opti_coordinate_to_camera_coordinate(point, transformation)
                cv2.circle(resized_frame, (int(camera_position[0]), int(camera_position[1])), 3, (0, 255, 0), -1)
            cv2.imshow("result", resized_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                new_data = {}

                new_data['camera'] = self.data['camera']

                new_data['roi_x'] = roi_x
                new_data['roi_y'] = roi_y
                new_data['roi_width'] = roi_width
                new_data['roi_height'] = roi_height
        
                new_data['scale_x'] = scale_x
                new_data['scale_y'] = scale_y
                new_data['translate_x'] = translate_x
                new_data['translate_y'] = translate_y

                self.update_calibration_data(new_data)
                break
        cap.release()

    def update_calibration_data(self, data):
        # write the updated data to the file
        with open(self.file_path, 'w') as file:
            json.dump(data, file, indent=2, cls=NumpyEncoder)

        print("Calibration finished! Data has been updated!")

    # For undistorting the image
    def undistort(self, frame):
        h, w = frame.shape[:2]
        mapx, mapy = cv2.initUndistortRectifyMap(self.cameraMatrix, self.dist, None, self.cameraMatrix, (w, h), 5)
        return cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

    # For ArUco
    def DetectArucoPose(self, frame):
        aruco_A_position = aruco_B_position = None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

        parameters = aruco.DetectorParameters_create()
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if ids is not None and len(ids) == 2:
            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, self.cameraMatrix, self.dist)
            # print("rvec: ", rvec)
            # print("tvex: ", tvec)
            for i in range(rvec.shape[0]):
                tvecCopy = tvec[i, :, :] + [10., 0, 0]
                # print("tvecCopy", tvecCopy)
                aruco.drawAxis(frame, self.cameraMatrix, self.dist, rvec[i, :, :], tvec[i, :, :], 0.03)
                aruco.drawDetectedMarkers(frame, corners, ids)
                # print("rvec[", i, ",: , ：]: ", rvec[i, :, :])

            for i in range(len(ids)):
                if ids[i] == self.aruco_A_id:
                    corner_A = corners[i][0]
                    # pixel position of marker A
                    center_x = (corner_A[0][0] + corner_A[1][0] + corner_A[2][0] + corner_A[3][0]) / 4
                    center_y = (corner_A[0][1] + corner_A[1][1] + corner_A[2][1] + corner_A[3][1]) / 4
                    aruco_A_position = [center_x, center_y]
                    print("aruco marker A position: ", [int(center_x), int(center_y)])
                    cv2.circle(frame, (int(center_x), int(center_y)), 3, (255, 0, 0), -1)

                if ids[i] == self.aruco_B_id:
                    corner_B = corners[i][0]
                    center_x = (corner_B[0][0] + corner_B[1][0] + corner_B[2][0] + corner_B[3][0]) / 4
                    center_y = (corner_B[0][1] + corner_B[1][1] + corner_B[2][1] + corner_B[3][1]) / 4
                    aruco_B_position = [center_x, center_y]
                    print("aruco marker B position: ", [int(center_x), int(center_y)])
                    cv2.circle(frame, (int(center_x), int(center_y)), 3, (0, 0, 255), -1)

            
        return frame, aruco_A_position, aruco_B_position

    # for optitrack
    def receiveData(self,value) -> None:
        self.mocap_data = value

    def parseArgs(self, arg_list, args_dict) -> dict:
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
        options_dict = self.parseArgs(sys.argv, options_dict)

        streaming_client = nnc.NatNetClient()
        streaming_client.set_client_address(options_dict["clientAddress"])
        streaming_client.set_server_address(options_dict["serverAddress"])
        streaming_client.set_use_multicast(options_dict["use_multicast"])

        streaming_client.mocap_data_listener = self.receiveData

        streaming_client.run()

    def cal_distance(self, A, B):
        distance = np.sqrt((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2)
        print("distance: ", distance)
        return distance

    def convert_opti_coordinate_to_camera_coordinate(self, opti_position, transformation):
        marker_pixel_postion_x = transformation[0] * opti_position[1] + transformation[2]
        marker_pixel_postion_y = transformation[1] * opti_position[0] + transformation[3]

        marker_pixel_postion = np.array([marker_pixel_postion_x, marker_pixel_postion_y])

        return marker_pixel_postion

    def convert_opti_coordinate_to_workspace(self, opti_position):
        '''
        The function to convert the optiTrack coordinate (-x, z, y) to the desired workspace coordinate
        Input: np.array[3]
        Output: the coordinate of the marker in workspace coordinate (x, y, z)
        '''
        workspace_position = np.array([-opti_position[0], opti_position[2], opti_position[1]])
        print("workspace_position: ", workspace_position)
        return workspace_position

    def read_camera_calibration_data(self):
        camera_data = self.data["camera"]

        fx = camera_data["fx"]
        fy = camera_data["fy"]
        cx = camera_data["cx"]
        cy = camera_data["cy"]
        k1 = camera_data["k1"]
        k2 = camera_data["k2"]
        p1 = camera_data["p1"]
        p2 = camera_data["p2"]
        k3 = camera_data["k3"]

        self.cameraMatrix = np.array([[fx, 0, cx],
                                 [0, fy, cy],
                                 [0, 0, 1]])
        self.dist = np.array([k1, k2, p1, p2, k3])


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

aligner = Aligner()
aligner.calibration()