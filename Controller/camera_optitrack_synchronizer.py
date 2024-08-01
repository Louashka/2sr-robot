import json
import numpy as np
import cv2
import threading
from datetime import datetime

class Aligner:
    def __init__(self) -> None:
        self.file_path = "Controller/calibration_data.json"
        self.data = None

        self.wait_video = False
        self.finish = False
        self.config_list = []

        try:
            with open(self.file_path, "r") as json_file:
                data = json.load(json_file)
                self.data = data
        except FileNotFoundError:
            print(f"Error: {self.file_path} not found.")
            return

        self.__read_camera_calibration_data()

    def startVideo(self, path_x, path_y, config0):
        self.config_list.append(config0)
        thread = threading.Thread(target=self.__run, args=(path_x, path_y))
        thread.start()

    def add_config(self, config) -> None:
        self.config_list.append(config)
        
    def __run(self, path_x, path_y):
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        video_path = f'Experiments/Video/path_tracking_{current_time}.mp4'

        cap = cv2.VideoCapture(0)
        # set the resolution to 1280x720
        cap.set(3, 1280)
        cap.set(4, 720)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 8.0, (1421,720))


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

            # map the trajectory to the coordinate
            for i in range(0, len(path_x)):
                point = np.array([path_x[i], path_y[i]])
                camera_position = self.__convert_opti_coordinate_to_camera_coordinate(point)
                cv2.circle(resized_frame, (int(camera_position[0]), int(camera_position[1])), 3, (0, 255, 0), -1)
            for i in range(len(self.config_list)):
                point = np.array(self.config_list[i][:2])
                camera_position = self.__convert_opti_coordinate_to_camera_coordinate(point)
                cv2.circle(resized_frame, (int(camera_position[0]), int(camera_position[1])), 3, (0, 0, 255), -1)
            
            out.write(resized_frame)
            cv2.imshow("RGB camera", resized_frame)
            self.wait_video = True

            if cv2.waitKey(1) & 0xFF == ord('q') or self.finish:
                self.finish = True
                break

        cap.release()
        out.release()

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

        marker_pixel_postion = np.array([marker_pixel_postion_x, marker_pixel_postion_y])

        return marker_pixel_postion
