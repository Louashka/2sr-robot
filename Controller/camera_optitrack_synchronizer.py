import json
import numpy as np
import cv2
import threading

class Aligner:
    def __init__(self) -> None:
        self.file_path = "Controller/calibration_data.json"
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
        self.wait_video = False

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
        self.__read_camera_calibration_data(camera_data)
        self.__read_aruco_calibration_data(aruco_data)
        self.__read_opti_calibration_data(opti_data)

    def startVideo(self, path_x, path_y):
        thread = threading.Thread(target=self.__run, args=(path_x, path_y))
        thread.start()
        
    def __run(self, path_x, path_y):
        cap = cv2.VideoCapture(0)
        fps = cap.get(5)
        # set the resolution to 1280x720
        cap.set(3, 1280)
        cap.set(4, 720)

        # ArUco Markers
        aruco_A_position = self.aruco_A_position
        aruco_B_position = self.aruco_B_position

        # OptiTrack Markers
        opti_A1_position = self.opti_A1_position
        opti_A2_position = self.opti_A2_position
        opti_B1_position = self.opti_B1_position
        opti_B2_position = self.opti_B2_position

        opti_A1_position = self.__convert_opti_coordinates_to_workspace(opti_A1_position)
        opti_A2_position = self.__convert_opti_coordinates_to_workspace(opti_A2_position)
        opti_B1_position = self.__convert_opti_coordinates_to_workspace(opti_B1_position)
        opti_B2_position = self.__convert_opti_coordinates_to_workspace(opti_B2_position)

        while cap.isOpened():
            _, frame = cap.read()
            undistort_frame = self.__undistort(frame)
            undistort_frame = cv2.rotate(undistort_frame, cv2.ROTATE_180)

            aruco_pixel_distance_x = aruco_B_position[0] - aruco_A_position[0]
            aruco_pixel_distance_y = aruco_B_position[1] - aruco_A_position[1]

            # Calculate the Region of Interest (ROI) based on the Aruco marker positions
            roi_x = int(min(aruco_A_position[0], aruco_B_position[0]))
            roi_y = int(min(aruco_A_position[1], aruco_B_position[1]))
            roi_width = int(abs(aruco_pixel_distance_x))
            roi_height = int(abs(aruco_pixel_distance_y))

            # Crop the frame using the calculated ROI
            cropped_frame = undistort_frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]

            # Resize the cropped frame while maintaining the aspect ratio
            height = 720
            width = int(self.frame_size[1])
            resized_frame = cv2.resize(cropped_frame, (width, height), interpolation=cv2.INTER_AREA)
            frame_size = self.frame_size

            # Display the resized frame
            # cv2.imshow('Resized Frame', resized_frame)
            opti_A_position = self.opti_A_position
            opti_B_position = self.opti_B_position

            coordinate_origin = opti_B_position

            markerA_pos = np.array(
                [opti_A_position[0] - coordinate_origin[0], opti_A_position[1] - coordinate_origin[1]])
            markerB_pos = np.array(
                [opti_B_position[0] - coordinate_origin[0], opti_B_position[1] - coordinate_origin[1]])  # (0, 0)

            # calculate the distance in meters and the pixel distance of two coordinates, and calculate the ratio
            aruco_distance = self.__calc_distance(frame_size, np.array([0, 0]))  # pixel
            opti_distance = self.__calc_distance(markerA_pos, markerB_pos)  # meter
            ratio = aruco_distance / opti_distance  # pixel to meter

            # convert the opti coordinate back to camera coordinate
            optiA_cam_pos = self.__convert_opti_coordinate_to_camera_coordinate(markerA_pos, frame_size, ratio)
            optiB_cam_pos = self.__convert_opti_coordinate_to_camera_coordinate(markerB_pos, frame_size, ratio)

            cv2.circle(resized_frame, (int(optiA_cam_pos[0]), int(optiA_cam_pos[1])), 30, (0, 255, 0), -1)
            cv2.circle(resized_frame, (int(optiB_cam_pos[0]), int(optiB_cam_pos[1])), 30, (0, 255, 0), -1)

            # map the trajectory to the coordinate
            for i in range(0, len(path_x)):
                point = np.array([path_y[i] - coordinate_origin[0], path_x[i] - coordinate_origin[1]])
                camera_position = self.__convert_opti_coordinate_to_camera_coordinate(point, frame_size, ratio)
                cv2.circle(resized_frame, (int(camera_position[0]), int(camera_position[1])), 3, (0, 255, 0), -1)
            cv2.imshow("RGB camera", resized_frame)
            self.wait_video = True

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()

    def __read_camera_calibration_data(self, camera):
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

    def __read_aruco_calibration_data(self, aruco):
        aruco_A_position = next((marker for marker in aruco["markers"] if marker['id'] == self.aruco_A_id), None)
        aruco_A_position = np.array(aruco_A_position["position"])
        aruco_B_position = next((marker for marker in aruco["markers"] if marker['id'] == self.aruco_B_id), None)
        aruco_B_position = np.array(aruco_B_position["position"])
        self.aruco_A_position = aruco_A_position
        self.aruco_B_position = aruco_B_position

    def __read_opti_calibration_data(self, opti):
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

    def __undistort(self, frame):
        h, w = frame.shape[:2]
        mapx, mapy = cv2.initUndistortRectifyMap(self.cameraMatrix, self.dist, None, self.cameraMatrix, (w, h), 5)
        return cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

    def __convert_opti_coordinates_to_workspace(self, opti_position):
        workspace_position = np.array([-opti_position[0], opti_position[2], opti_position[1]])
        return workspace_position
    
    def __calc_distance(self, A, B):
        distance = np.sqrt((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2)
        return distance
    
    def __convert_opti_coordinate_to_camera_coordinate(self, opti_position, frame_size, ratio):
        aruco_A_to_B_x = frame_size[1]
        aruco_A_to_B_y = frame_size[0]
        pixel_position = opti_position * ratio  # directly convert the meter length to pixel length
        # print("pixel_position: ", pixel_position)
        marker_pixel_postion_x = aruco_A_to_B_x - pixel_position[1]
        marker_pixel_postion_y = aruco_A_to_B_y - pixel_position[0]
        marker_pixel_postion = np.array([marker_pixel_postion_x, marker_pixel_postion_y])

        return marker_pixel_postion
