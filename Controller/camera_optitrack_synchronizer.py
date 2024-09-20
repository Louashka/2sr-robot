import json
import numpy as np
import cv2
import threading

from Model import global_var as gv

class Aligner:
    def __init__(self) -> None:
        self.file_path = "Controller/calibration_data.json"
        self.data = None

        self.wait_video = False
        self.finish = False
        self.config_list = []

        self.offset = [-15, -3]

        try:
            with open(self.file_path, "r") as json_file:
                data = json.load(json_file)
                self.data = data
        except FileNotFoundError:
            print(f"Error: {self.file_path} not found.")
            return

        self.__read_camera_calibration_data()

    def startVideo(self, config_target: list, config0, date_title):
        self.config_list.append(config0)
        thread = threading.Thread(target=self.__run, args=(config_target, date_title))
        thread.start()

    def add_config(self, config) -> None:
        self.config_list.append(config)
        
    def __run(self, config_target: list, date_title: str):
        video_path_rgb = f'Experiments/Video/Tracking/SM1/sm1_rgb_{date_title}.mp4'
        video_path_thermal = f'Experiments/Video/Tracking/SM1/sm1_thermal_{date_title}.mp4'

        target_seg1 = self.__arc(config_target, 1)
        target_seg2 = self.__arc(config_target, 2)

        cap_rgb = cv2.VideoCapture(1)
        # set the resolution to 1280x720
        cap_rgb.set(3, 1280)
        cap_rgb.set(4, 720)

        cap_thermal = cv2.VideoCapture(0)
        # set the resolution to 1280x720
        cap_thermal.set(3, 1280)
        cap_thermal.set(4, 720)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_rgb = cv2.VideoWriter(video_path_rgb, fourcc, 16.0, (1409,720))
        out_thermal = cv2.VideoWriter(video_path_thermal, fourcc, 16.0, (640,512))


        while cap_rgb.isOpened():
            _, frame = cap_rgb.read()
            undistort_frame = self.__undistort(frame)
            undistort_frame = cv2.rotate(undistort_frame, cv2.ROTATE_180)

             # Crop the frame using the calculated ROI
            cropped_frame = undistort_frame[self.data['roi_y']:self.data['roi_y'] + self.data['roi_height'], 
                                            self.data['roi_x']:self.data['roi_x'] + self.data['roi_width']]

            # Resize the cropped frame while maintaining the aspect ratio
            height = 720
            width = int(cropped_frame.shape[1] * (height / cropped_frame.shape[0]))
            resized_frame = cv2.resize(cropped_frame, (width, height), interpolation=cv2.INTER_AREA)

            cv2.polylines(resized_frame, [target_seg1], False, (255, 217, 4), 2)
            cv2.polylines(resized_frame, [target_seg2], False, (255, 217, 4), 2)

            # map the trajectory to the coordinate
            # for i in range(0, path.n):
            #     point = np.array([path.x[i], path.y[i]])
            #     camera_position = self.__convert_opti_coordinate_to_camera_coordinate(point)
            #     cv2.circle(resized_frame, (camera_position[0], camera_position[1]), 3, (255, 217, 4), -1)
            # for i in range(len(self.config_list)):
            #     point = np.array(self.config_list[i][:2])
            #     camera_position = self.__convert_opti_coordinate_to_camera_coordinate(point)
            #     cv2.circle(resized_frame, (camera_position[0], camera_position[1]), 3, (49, 49, 255), -1)
            
            out_rgb.write(resized_frame)
            cv2.imshow("RGB camera", resized_frame)

            _, frame_thermal = cap_thermal.read()
            frame_thermal = cv2.rotate(frame_thermal, cv2.ROTATE_180)
            out_thermal.write(frame_thermal)
            cv2.imshow("Thermal camera", frame_thermal)

            self.wait_video = True

            if cv2.waitKey(1) & 0xFF == ord('q') or self.finish:
                self.finish = True
                break

        cap_rgb.release()
        out_rgb.release()

        cap_thermal.release()
        out_thermal.release()

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
        th0 = config[2] - 0.0
        k = config[2+seg]
        # if seg == 2:
        #     k = 0
        l = np.linspace(0, gv.L_VSS, 10)

        flag = -1 if seg == 1 else 1
        theta_array = th0 + flag * k * l

        if k == 0:
            x = np.array([0, flag * gv.L_VSS * np.cos(th0)])
            y = np.array([0, flag * gv.L_VSS * np.sin(th0)])
        else:
            x = np.sin(theta_array) / k - np.sin(th0) / k
            y = -np.cos(theta_array) / k + np.cos(th0) / k

        x += config[0]
        y += config[1]

        points = []
        for x_i, y_i in zip(x, y):
            point = self.__convert_opti_coordinate_to_camera_coordinate([x_i, y_i])
            points.append(point)

        points = np.array(points).reshape((-1, 1, 2))

        return points

