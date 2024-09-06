import json
import numpy as np
import cv2
import threading

from Model import splines, global_var as gv

class Aligner:
    def __init__(self) -> None:
        self.file_path = "Controller/calibration_data.json"
        self.data = None

        self.wait_video = False
        self.finish = False
        self.config_list = []

        self.offset = [-28, -20]

        try:
            with open(self.file_path, "r") as json_file:
                data = json.load(json_file)
                self.data = data
        except FileNotFoundError:
            print(f"Error: {self.file_path} not found.")
            return

        self.__read_camera_calibration_data()

    def startVideo(self, path: splines.TrajectoryShape, config0, date_title):
        self.config_list.append(config0)
        thread = threading.Thread(target=self.__run, args=(path, date_title))
        thread.start()

    def add_config(self, config) -> None:
        self.config_list.append(config)
        
    def __run(self, path: splines.TrajectoryShape, date_title: str):
        video_path = f'Experiments/Video/soft_mode_test_{date_title}.mp4'

        target_seg1 = self.__arc(path.getPoint(path.n-1), 1)
        target_seg2 = self.__arc(path.getPoint(path.n-1), 2)

        cap = cv2.VideoCapture(0)
        # set the resolution to 1280x720
        cap.set(3, 1280)
        cap.set(4, 720)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 8.0, (1409,720))


        while cap.isOpened():
            _, frame = cap.read()
            undistort_frame = self.__undistort(frame)
            undistort_frame = cv2.rotate(undistort_frame, cv2.ROTATE_180)

             # Crop the frame using the calculated ROI
            cropped_frame = undistort_frame[self.data['roi_y']:self.data['roi_y'] + self.data['roi_height'], 
                                            self.data['roi_x']:self.data['roi_x'] + self.data['roi_width']]

            # Resize the cropped frame while maintaining the aspect ratio
            height = 720
            width = int(cropped_frame.shape[1] * (height / cropped_frame.shape[0]))
            resized_frame = cv2.resize(cropped_frame, (width, height), interpolation=cv2.INTER_AREA)

            cv2.polylines(resized_frame, [target_seg1], False, (0, 255, 0), 2)
            cv2.polylines(resized_frame, [target_seg2], False, (0, 255, 0), 2)

            # map the trajectory to the coordinate
            for i in range(0, path.n):
                point = np.array([path.x[i], path.y[i]])
                camera_position = self.__convert_opti_coordinate_to_camera_coordinate(point)
                cv2.circle(resized_frame, (camera_position[0], camera_position[1]), 3, (0, 255, 0), -1)
            for i in range(len(self.config_list)):
                point = np.array(self.config_list[i][:2])
                camera_position = self.__convert_opti_coordinate_to_camera_coordinate(point)
                cv2.circle(resized_frame, (camera_position[0], camera_position[1]), 3, (0, 0, 255), -1)
            
            out.write(resized_frame)
            cv2.imshow("RGB camera", resized_frame)
            self.wait_video = True

            if cv2.waitKey(1) & 0xFF == ord('q') or self.finish:
                self.finish = True
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

    def __read_camera_calibration_data(self):
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

    def __undistort(self, frame):
        h, w = frame.shape[:2]
        mapx, mapy = cv2.initUndistortRectifyMap(self.cameraMatrix, self.dist, None, self.cameraMatrix, (w, h), 5)
        return cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
    
    def __convert_opti_coordinate_to_camera_coordinate(self, opti_position):
        marker_pixel_postion_x = self.data['scale_x'] * opti_position[1] + self.data['translate_x']
        marker_pixel_postion_y = self.data['scale_y'] * opti_position[0] + self.data['translate_y']

        marker_pixel_postion = [int(marker_pixel_postion_x) + self.offset[0], 
                                int(marker_pixel_postion_y) + self.offset[1]]

        return marker_pixel_postion
    
    def __arc(self, config, seg=1):
        l = np.linspace(0, gv.L_VSS, 10)
        flag = -1 if seg == 1 else 1
        theta_array = config[2] + flag * config[2+seg] * l

        if config[2+seg] == 0:
            x = np.array([0, flag * gv.L_VSS * np.cos(config[2])])
            y = np.array([0, flag * gv.L_VSS * np.sin(config[2])])
        else:
            x = np.sin(theta_array) / config[2+seg] - np.sin(config[2]) / config[2+seg]
            y = -np.cos(theta_array) / config[2+seg] + np.cos(config[2]) / config[2+seg]

        x += config[0]
        y += config[1]

        points = []
        for x_i, y_i in zip(x, y):
            point = self.__convert_opti_coordinate_to_camera_coordinate([x_i, y_i])
            points.append(point)

        points = np.array(points).reshape((-1, 1, 2))

        return points

