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
path_x = np.arange(0, 2, 0.01)
path_y = np.array([np.sin(x / 0.21) * x / 2.7 for x in path_x])
# shift
path_x = - path_x + 1
path_y = path_y - 0.3
# print("path x: ", path_x)
# print("path y: ", path_y)

class Aligner():

    def __init__(self):
        self.file_path = "Valens_Bugs\calibration_data.json"
        self.data = None
        self.cameraMatrix = None
        self.dist = None
        self.frame_size = None

        self.aruco_A_id = 10
        self.aruco_B_id = 12
        self.aruco_A_position = None
        self.aruco_B_position = None

        self.opti_A1_position = None
        self.opti_A2_position = None
        self.opti_B1_position = None
        self.opti_B2_position = None
        self.opti_A_position = None
        self.opti_B_position = None

        self.frame = None

        try:
            with open(self.file_path, "r") as json_file:
                data = json.load(json_file)
                self.data = data
        except FileNotFoundError:
            print(f"Error: {self.file_path} not found.")
            return

        camera_data = data["camera"]
        aruco_data = data["aruco"]
        opti_data = data["opti_track"]
        self.read_camera_calibration_data(camera_data)
        self.read_aruco_calibration_data(aruco_data)
        self.read_opti_calibration_data(opti_data)

    def run(self):
        cap = cv2.VideoCapture(0)
        fps = cap.get(5)
        # set the resolution to 1280x720
        cap.set(3, 1280)
        cap.set(4, 720)
        crop_flag = False

        # ArUco Markers
        aruco_A_id = self.aruco_A_id
        aruco_B_id = self.aruco_B_id
        aruco_A_position = self.aruco_A_position
        aruco_B_position = self.aruco_B_position

        # OptiTrack Markers
        opti_A1_position = self.opti_A1_position
        print("opti_A1_position: ", opti_A1_position)
        opti_A2_position = self.opti_A2_position
        opti_B1_position = self.opti_B1_position
        opti_B2_position = self.opti_B2_position

        opti_A1_position = self.convert_opti_coordinate_to_workspace(opti_A1_position)
        opti_A2_position = self.convert_opti_coordinate_to_workspace(opti_A2_position)
        opti_B1_position = self.convert_opti_coordinate_to_workspace(opti_B1_position)
        opti_B2_position = self.convert_opti_coordinate_to_workspace(opti_B2_position)

        while cap.isOpened():
            ret, frame = cap.read()
            h1, w1 = frame.shape[:2]
            undistort_frame = self.undistort(frame)
            undistort_frame = cv2.rotate(undistort_frame, cv2.ROTATE_180)

            frame_wait_for_crop, aruco_A_position, aruco_B_position = self.DetectArucoPose(undistort_frame, aruco_A_position, aruco_B_position)
            aruco_pixel_distance_x = aruco_B_position[0] - aruco_A_position[0]
            aruco_pixel_distance_y = aruco_B_position[1] - aruco_A_position[1]

            # Calculate the Region of Interest (ROI) based on the Aruco marker positions
            roi_x = int(min(aruco_A_position[0], aruco_B_position[0]))
            roi_y = int(min(aruco_A_position[1], aruco_B_position[1]))
            roi_width = int(abs(aruco_pixel_distance_x))
            roi_height = int(abs(aruco_pixel_distance_y))

            # Crop the frame using the calculated ROI
            cropped_frame = undistort_frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]
            print("cropped frame: ", cropped_frame)
            # print(cropped_frame.shape)

            # Resize the cropped frame while maintaining the aspect ratio
            height = 720
            width = int(self.frame_size[1])
            resized_frame = cv2.resize(cropped_frame, (width, height), interpolation=cv2.INTER_AREA)
            frame_shape = resized_frame.shape
            frame_size = self.frame_size
            print("resize frame shape: ", frame_size)

            # Display the resized frame
            cv2.imshow('Resized Frame', resized_frame)
            opti_A_position = self.opti_A_position
            opti_B_position = self.opti_B_position
            print("optiTrack marker set A: ", opti_A_position)
            print("optiTrack marker set B: ", opti_B_position)

            '''In case of being confused, here is a graph of how to convert the coordinate
            . ------> x (camera)
            |       -----------------------------------------
            |       | A                                     |
            v       |                         x (workspace)^|
    y (camera)      |                                      ||
                    |                    y(workspace)<----.B|
                    -----------------------------------------

            workspace origin (in meter) = x_position of marker B, y_position of marker B
            use the function '' to convert
            '''

            coordinate_origin = opti_B_position
            print("coordinate_origin: ", coordinate_origin)
            # print("markerA_pos - camera_zero: ", markerA_pos - camera_zero)
            markerA_pos = np.array(
                [opti_A_position[0] - coordinate_origin[0], opti_A_position[1] - coordinate_origin[1]])
            markerB_pos = np.array(
                [opti_B_position[0] - coordinate_origin[0], opti_B_position[1] - coordinate_origin[1]])  # (0, 0)
            print("markerA_pos: ", markerA_pos)
            print("markerB_pos: ", markerB_pos)

            # calculate the distance in meters and the pixel distance of two coordinates, and calculate the ratio
            aruco_distance = self.cal_distance(frame_size, np.array([0, 0]))  # pixel
            opti_distance = self.cal_distance(markerA_pos, markerB_pos)  # meter

            ratio = aruco_distance / opti_distance  # pixel to meter
            print("Ratio： ", ratio)
            # convert the opti coordinate back to camera coordinate
            optiA_cam_pos = self.convert_opti_coordinate_to_camera_coordinate(markerA_pos, frame_size, ratio)
            optiB_cam_pos = self.convert_opti_coordinate_to_camera_coordinate(markerB_pos, frame_size, ratio)
            print("optiA_cam_pos: ", optiA_cam_pos)
            print("optiB_cam_pos: ", optiB_cam_pos)
            cv2.circle(resized_frame, (int(optiA_cam_pos[0]), int(optiA_cam_pos[1])), 30, (0, 255, 0), -1)
            cv2.circle(resized_frame, (int(optiB_cam_pos[0]), int(optiB_cam_pos[1])), 30, (0, 255, 0), -1)
            # cv2.imshow("result", resized_frame)

            # map the trajectory to the coordinate
            for i in range(0, len(path_x)):
                point = np.array([path_y[i] - coordinate_origin[0], path_x[i] - coordinate_origin[1]])
                camera_position = self.convert_opti_coordinate_to_camera_coordinate(point, frame_size, ratio)
                cv2.circle(resized_frame, (int(camera_position[0]), int(camera_position[1])), 3, (0, 255, 0), -1)
            cv2.imshow("result", resized_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindwos()

    def calibration(self):
        # ArUco Markers
        aruco_A_id = self.aruco_A_id
        aruco_B_id = self.aruco_B_id
        aruco_A_position = self.aruco_A_position
        aruco_B_position = self.aruco_B_position

        frame_size = self.frame_size

        # OptiTrack Markers
        opti_A1_id = 27373
        opti_A2_id = 27379
        opti_B1_id = 27673
        opti_B2_id = 27674
        opti_A1_position = self.opti_A1_position
        opti_A2_position = self.opti_A2_position
        opti_B1_position = self.opti_B1_position
        opti_B2_position = self.opti_B2_position

        opti_A1_position = self.convert_opti_coordinate_to_workspace(opti_A1_position)
        opti_A2_position = self.convert_opti_coordinate_to_workspace(opti_A2_position)
        opti_B1_position = self.convert_opti_coordinate_to_workspace(opti_B1_position)
        opti_B2_position = self.convert_opti_coordinate_to_workspace(opti_B2_position)

        opti_A_position = np.array([(opti_A1_position[0]+opti_A2_position[0]) / 2,
                                    (opti_A1_position[0]+opti_A2_position[0]) / 2])      # mid-point of the marker A under opti coordinate
        opti_B_position = np.array([(opti_B1_position[0]+opti_B2_position[0]) / 2,
                                    (opti_B1_position[0]+opti_B2_position[0]) / 2])      # mid-point of the marker B under opti coordinate

        cap = cv2.VideoCapture(0)
        width = cap.get(3)
        height = cap.get(4)
        fps = cap.get(5)
        # print(width, height, fps)  # 640.0 480.0 30.0

        # set the resoulution to 1280x720
        cap.set(3, 1280)
        cap.set(4, 720)
        width = cap.get(3)
        height = cap.get(4)
        print(width, height, fps)  # 1280.0 720.0 30.0

        crop_flag = False

        self.startDataListener()

        while cap.isOpened():
            ret, frame = cap.read()
            h1, w1 = frame.shape[:2]
            undistort_frame = self.undistort(frame)
            undistort_frame = cv2.rotate(undistort_frame, cv2.ROTATE_180)
            # cv2.imshow("rotated by 180", rotated_frame)

            frame_wait_for_crop, aruco_A_position, aruco_B_position = self.DetectArucoPose(undistort_frame, aruco_A_position, aruco_B_position)
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
            frame_shape = resized_frame.shape
            frame_size = np.array([frame_shape[0], frame_shape[1]])
            print("resize frame shape: ", frame_size)

            # Display the resized frame
            cv2.imshow('Resized Frame', resized_frame)

            opti_A_position[0] = (opti_A1_position[0] + opti_A2_position[0]) / 2
            opti_A_position[1] = (opti_A1_position[1] + opti_A2_position[1]) / 2

            opti_B_position[0] = (opti_B1_position[0] + opti_B2_position[0]) / 2
            opti_B_position[1] = (opti_B1_position[1] + opti_B2_position[1]) / 2
            print("optiTrack marker set A: ", opti_A_position)
            print("optiTrack marker set B: ", opti_B_position)
            
            '''In case of being confused, here is a graph of how to convert the coordinate
            . ------> x (camera)
            |       -----------------------------------------
            |       | A                                     |
            v       |                         x (workspace)^|
    y (camera)      |                                      ||
                    |                    y(workspace)<----.B|
                    -----------------------------------------

            workspace origin (in meter) = x_position of marker B, y_position of marker B
            use the function '' to convert
            '''

            coordinate_origin = opti_B_position
            print("coordinate_origin: ", coordinate_origin)
            # print("markerA_pos - camera_zero: ", markerA_pos - camera_zero)
            markerA_pos = np.array([opti_A_position[0] - coordinate_origin[0], opti_A_position[1] - coordinate_origin[1]])
            markerB_pos = np.array([opti_B_position[0] - coordinate_origin[0], opti_B_position[1] - coordinate_origin[1]])  # (0, 0)
            print("markerA_pos: ", markerA_pos)
            print("markerB_pos: ", markerB_pos)

            # calculate the distance in meters and the pixel distance of two coordinates, and calculate the ratio
            aruco_distance = self.cal_distance(frame_size, np.array([0, 0]))     # pixel
            opti_distance = self.cal_distance(markerA_pos, markerB_pos)          # meter

            ratio = aruco_distance / opti_distance      # pixel to meter
            print("Ratio： ", ratio)
            # convert the opti coordinate back to camera coordinate
            optiA_cam_pos = self.convert_opti_coordinate_to_camera_coordinate(markerA_pos, frame_size, ratio)
            optiB_cam_pos = self.convert_opti_coordinate_to_camera_coordinate(markerB_pos, frame_size, ratio)
            print("optiA_cam_pos: ", optiA_cam_pos)
            print("optiB_cam_pos: ", optiB_cam_pos)
            cv2.circle(resized_frame, (int(optiA_cam_pos[0]), int(optiA_cam_pos[1])), 30, (0, 255, 0), -1)
            cv2.circle(resized_frame, (int(optiB_cam_pos[0]), int(optiB_cam_pos[1])), 30, (0, 255, 0), -1)
            # cv2.imshow("result", resized_frame)

            # map the trajectory to the coordinate
            for i in range (0, len(path_x)):
                point = np.array([path_y[i] - coordinate_origin[0], path_x[i] - coordinate_origin[1]])
                camera_position = self.convert_opti_coordinate_to_camera_coordinate(point, frame_size, ratio)
                cv2.circle(resized_frame, (int(camera_position[0]), int(camera_position[1])), 3, (0, 255, 0), -1)
            cv2.imshow("result", resized_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.aruco_A_position = aruco_A_position
                self.aruco_B_position = aruco_B_position
                self.opti_A1_position = opti_A1_position
                self.opti_A2_position = opti_A2_position
                self.opti_B1_position = opti_B1_position
                self.opti_B2_position = opti_B2_position
                self.frame_size = frame_size
                self.update_calibration_data()
                break
        cap.release()
        cv2.destroyAllWindwos()

    def update_calibration_data(self):
        data = self.data
        # Update the camera frame size
        data["camera"]['frame_size'] = self.frame_size

        # update the aruco marker position
        for marker in data["aruco"]["markers"]:
            print(marker)
            if marker["id"] == self.aruco_A_id:
                marker["position"] = self.aruco_A_position
            elif marker["id"] == self.aruco_B_id:
                marker["position"] = self.aruco_A_position

        # update the opti marker position
        data["opti_track"]["opti_A1_position"] = self.opti_A1_position
        data["opti_track"]["opti_A2_position"] = self.opti_A2_position
        data["opti_track"]["opti_B1_position"] = self.opti_B1_position
        data["opti_track"]["opti_B2_position"] = self.opti_B2_position

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
    def DetectArucoPose(self, frame, aruco_A_position, aruco_B_position):

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

        parameters = aruco.DetectorParameters_create()
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

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
                    markerA_position = [center_x, center_y]
                    aruco_A_position = markerA_position
                    print("aruco marker A position: ", [int(center_x), int(center_y)])
                    cv2.circle(frame, (int(center_x), int(center_y)), 3, (0, 0, 255), -1)

                if ids[i] == self.aruco_B_id:
                    corner_B = corners[i][0]
                    center_x = (corner_B[0][0] + corner_B[1][0] + corner_B[2][0] + corner_B[3][0]) / 4
                    center_y = (corner_B[0][1] + corner_B[1][1] + corner_B[2][1] + corner_B[3][1]) / 4
                    markerB_position = [center_x, center_y]
                    aruco_B_position = markerB_position
                    print("aruco marker B position: ", [int(center_x), int(center_y)])
                    cv2.circle(frame, (int(center_x), int(center_y)), 3, (0, 0, 255), -1)

            # print("Ratio: ", ratio)
        return frame, aruco_A_position, aruco_B_position

    # for optitrack
    def receiveData(self,value) -> None:
        global data
        data = value

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

    def unpackData(self):
        if data is not None:

            labeled_marker_data = data.labeled_marker_data

            labeled_marker_list = labeled_marker_data.labeled_marker_list

            for marker in labeled_marker_list:
                model_id, marker_id = [int(i) for i in marker.get_id()]
                marker_info = {'model_id': model_id, 'marker_id': marker_id,
                               'marker_x': marker.pos[0], 'marker_y': marker.pos[1], 'marker_z': marker.pos[2]}
                print(marker_info)

                if marker_id == self.opti_A1_id:
                    opti_position = marker.pos
                    opti_A1_position = self.convert_opti_coordinate_to_workspace(opti_position)
                    self.opti_A1_position = opti_A1_position
                    print("OptiTrack Marker A1 position: ", opti_A1_position)

                if marker_id == self.opti_A2_id:
                    opti_position = marker.pos
                    opti_A2_position = self.convert_opti_coordinate_to_workspace(opti_position)
                    self.opti_A2_position = opti_A2_position
                    print("OptiTrack Marker A2 position: ", opti_A2_position)

                if marker_id == self.opti_B1_id:
                    opti_position = marker.pos
                    opti_B1_position = self.convert_opti_coordinate_to_workspace(opti_position)
                    self.opti_B1_position = opti_B1_position
                    print("OptiTrack Marker B1 position: ", opti_B1_position)

                if marker_id == self.opti_B2_id:
                    opti_position = marker.pos
                    opti_B2_position = self.convert_opti_coordinate_to_workspace(opti_position)
                    self.opti_B2_position = opti_B2_position
                    print("OptiTrack Marker B2 position: ", opti_B2_position)

    def cal_distance(self, A, B):
        distance = np.sqrt((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2)
        print("distance: ", distance)
        return distance

    def convert_opti_coordinate_to_camera_coordinate(self, opti_position, frame_size, ratio):
        '''
        Input:
            opti_position: the position of the point under the opti coordinate, and has been convert from the
                            global coordinate to the coordinate whose origin is marker B
            frame_size: the size of the frame, (height, width) = (720, ?)
            ratio: the ratio of converting meter to pixel, which should be calculated based on the distance
                            (both pixel distance and the meter distance) of two markers
        Output: the camera frame position refer to marker A
        '''
        aruco_A_to_B_x = frame_size[1]
        aruco_A_to_B_y = frame_size[0]
        pixel_position = opti_position * ratio  # directly convert the meter length to pixel length
        # print("pixel_position: ", pixel_position)
        marker_pixel_postion_x = aruco_A_to_B_x - pixel_position[1]
        marker_pixel_postion_y = aruco_A_to_B_y - pixel_position[0]
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

    def read_camera_calibration_data(self, camera):
        fx = camera["fx"]
        fy = camera["fy"]
        cx = camera["cx"]
        cy = camera["cy"]
        k1 = camera["k1"]
        k2 = camera["k2"]
        p1 = camera["p1"]
        p2 = camera["p2"]
        k3 = camera["k3"]
        frame_size = camera["frame_size"]
        self.cameraMatrix = np.array([[fx, 0, cx],
                                 [0, fy, cy],
                                 [0, 0, 1]])
        self.dist = np.array([k1, k2, p1, p2, k3])
        self.frame_size = frame_size

    def read_aruco_calibration_data(self, aruco):
        aruco_A_position = next((marker for marker in aruco["markers"] if marker['id'] == self.aruco_A_id), None)
        aruco_A_position = np.array(aruco_A_position["position"])
        aruco_B_position = next((marker for marker in aruco["markers"] if marker['id'] == self.aruco_B_id), None)
        aruco_B_position = np.array(aruco_B_position["position"])
        self.aruco_A_position = aruco_A_position
        self.aruco_B_position = aruco_B_position

    def read_opti_calibration_data(self, opti):
        opti_A1_position = opti["opti_A1_position"]
        opti_A2_position = opti["opti_A2_position"]
        opti_B1_position = opti["opti_B1_position"]
        opti_B2_position = opti["opti_B2_position"]

        opti_A_position = np.array([
            (opti_A1_position[0] + opti_A2_position[0]) / 2,
            (opti_A1_position[1] + opti_A2_position[1]) / 2,
            (opti_A1_position[2] + opti_A2_position[2]) / 2
        ])
        # opti_A_position = self.convert_opti_coordinate_to_workspace(opti_A_position)

        opti_B_position = np.array([
            (opti_B1_position[0] + opti_B2_position[0]) / 2,
            (opti_B1_position[1] + opti_B2_position[1]) / 2,
            (opti_B1_position[2] + opti_B2_position[2]) / 2
        ])
        # opti_B_position = self.convert_opti_coordinate_to_workspace(opti_B_position)

        self.opti_A_position = opti_A_position
        self.opti_B_position = opti_B_position
        self.opti_A1_position = opti_A1_position
        self.opti_A2_position = opti_A2_position
        self.opti_B1_position = opti_B1_position
        self.opti_B2_position = opti_B2_position

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

aligner = Aligner()
# aligner.run()
aligner.calibration()