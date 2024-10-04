import json
import numpy as np
import cv2
import threading
import time

from Model import global_var as gv

class Aligner:
    def __init__(self) -> None:
        self.file_path = "Controller/calibration_data.json"
        self.data = None

        self.wait_video = False
        self.finish = False

        self.config_list = []
        
        self.current_config = None
        self.markers = None
        self.circle_shape = None
        self.cheescake_contour = None
        self.contact_point = None

        try:
            with open(self.file_path, "r") as json_file:
                data = json.load(json_file)
                self.data = data
        except FileNotFoundError:
            print(f"Error: {self.file_path} not found.")
            return

        self.__read_camera_calibration_data()

    def add_config(self, config) -> None:
        self.config_list.append(config)
    
    # def startVideo(self, config_target: list, config0, date_title):
    #     self.config_list.append(config0)
    #     thread = threading.Thread(target=self.__run, args=(config_target, date_title))
    #     thread.start()

    def startVideo(self, date_title: str, task: str, args=[]):
        if task == 'soft_modes':
            config_target, config0 = args
            self.config_list.append(config0)
            thread = threading.Thread(target=self.__run, args=(config_target, date_title))
            thread.start()
        elif task == 'object_handling':
            thread = threading.Thread(target=self.__runOH, args=(date_title,))
            thread.start()

    def __runOH(self, date_title: str):
        video_path_rgb = f'Experiments/Video/Grasping/grasp_{date_title}.mp4'

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path_rgb, fourcc, 16.0, (1080,520))

        cap_rgb = cv2.VideoCapture(0)
        # set the resolution to 1280x720
        cap_rgb.set(3, 1280)
        cap_rgb.set(4, 720)

        cheescake_pos = (0, 0)
        cheescake_r = 0

        start_timer = False
        time_start = time.perf_counter()
        elapsed_time = 0

        while cap_rgb.isOpened():
            _, frame = cap_rgb.read()
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            h, w = frame.shape[:2]
            self.new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist, (w,h), 1, (w,h))
            undistorted_frame = cv2.undistort(frame, self.camera_matrix, self.dist, None, self.new_camera_matrix)

            gray = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2GRAY)
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100)
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                self.circle_shape = circles
            if self.circle_shape is not None:
                for (x, y, r) in self.circle_shape:
                    cv2.circle(undistorted_frame, (x, y), r, (255, 217, 4), 2)
                    cheescake_pos = (x, y)
                    cheescake_r = r

            mean_z = 0
            # cheescake_depth = 0

            if self.markers is not None:
                z_values = [marker['marker_z'] for marker in self.markers.values()]
                mean_z = sum(z_values) / len(z_values)

                # depth_list = []

                # for marker in self.markers.values():
                #     image_point, depth = self.globalToImage(marker['marker_x'], marker['marker_y'], marker['marker_z'])
                    # depth_list.append(depth)
                    # Draw the point on the image
                    # cv2.circle(undistorted_frame, image_point, 3, (0, 0, 255), -1)

                # cheescake_depth = sum(depth_list) / len(depth_list)

                # self.cheescake_pos_global = self.imageToGlobal(cheescake_pos, cheescake_depth)
                # self.cheescake_contour = []

                # theta = np.linspace(0, 2*np.pi, 30)
                # cheescake_x = cheescake_pos[0] + cheescake_r * np.cos(theta)
                # cheescake_y = cheescake_pos[1] + cheescake_r * np.sin(theta)

                # for x, y in zip(cheescake_x, cheescake_y):
                #     self.cheescake_contour.append(self.imageToGlobal((x, y), cheescake_depth))

            if self.current_config is not None:
                seg1 = self.__arc(self.current_config, mean_z, 1)
                seg2 = self.__arc(self.current_config, mean_z, 2)

                cv2.polylines(undistorted_frame, [seg1], False, (255, 217, 4), 2)
                cv2.polylines(undistorted_frame, [seg2], False, (255, 217, 4), 2)

            if self.contact_point is not None:
                cp_image, _ = self.globalToImage(*self.contact_point, mean_z)
                cv2.circle(undistorted_frame, cp_image, 3, (0, 0, 255), -1)

            # Crop undistorted_frame from all sides
            h, w = undistorted_frame.shape[:2]
            crop_margin = 100  # Adjust this value to increase or decrease the crop amount
            cropped_frame = undistorted_frame[crop_margin:h-crop_margin, crop_margin:w-crop_margin]

            cv2.imshow("RGB camera", cropped_frame)
            out.write(cropped_frame)

            self.wait_video = True

            if cv2.waitKey(1) & 0xFF == ord('q') or self.finish:
                self.finish = True
                if not start_timer:
                    time_start = time.perf_counter()
                start_timer = True

            if start_timer:
                if elapsed_time > 3:
                    break

                elapsed_time = time.perf_counter() - time_start


        cap_rgb.release()
        out.release()

        cv2.destroyAllWindows()

        
    def __run(self, config_target: list, date_title: str):
        video_path_rgb = f'Experiments/Video/Tracking/SM2/sm2_rgb_{date_title}.mp4'
        video_path_thermal = f'Experiments/Video/Tracking/SM2/sm2_thermal_{date_title}.mp4'

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
        self.camera_matrix = np.array([[fx, 0, cx],
                                 [0, fy, cy],
                                 [0, 0, 1]])
        self.dist = np.array([k1, k2, p1, p2, k3])
        self.R = np.array(self.data["R"])
        self.tvec = np.array(self.data["tvec"]).reshape(3,1)

    
    def globalToCamera(self, point_global):
        point_global = np.array(point_global).reshape(3, 1)
        point_camera = self.R @ point_global + self.tvec
        return point_camera.flatten()

    def cameraToImage(self, point_camera):
        x, y, z = point_camera
        u = self.new_camera_matrix[0, 0] * x / z + self.new_camera_matrix[0, 2]
        v = self.new_camera_matrix[1, 1] * y / z + self.new_camera_matrix[1, 2]
        return (int(u), int(v)), z
    
    def globalToImage(self, x, y, z):
        global_point = np.array([x, y, z]) 
        camera_point = self.globalToCamera(global_point)
        image_point, z = self.cameraToImage(camera_point)

        return image_point, z
    
    def imageToCamera(self, image_point, depth):
        """
        Convert image coordinates to camera coordinates.
        
        :param image_point: (u, v) coordinates in the image
        :param depth: The depth (Z coordinate) of the point in camera space
        :return: 3D point in camera coordinates
        """
        u, v = image_point
        fx = self.new_camera_matrix[0, 0]
        fy = self.new_camera_matrix[1, 1]
        cx = self.new_camera_matrix[0, 2]
        cy = self.new_camera_matrix[1, 2]

        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth

        return np.array([x, y, z])

    def cameraToGlobal(self, point_camera):
        """
        Convert camera coordinates to global coordinates.
        
        :param point_camera: 3D point in camera coordinates
        :return: 3D point in global coordinates
        """
        point_camera = np.array(point_camera).reshape(3, 1)
        point_global = np.linalg.inv(self.R) @ (point_camera - self.tvec)
        return point_global.flatten()
    
    def imageToGlobal(self, image_point, depth):
        """
        Convert image coordinates to global coordinates.
        
        :param image_point: (u, v) coordinates in the image
        :param depth: The depth (Z coordinate) of the point in camera space
        :return: 3D point in global coordinates
        """
        point_camera = self.imageToCamera(image_point, depth)
        point_global = self.cameraToGlobal(point_camera)
        return point_global
    
    def __arc(self, config, z, seg=1):
        th0 = config[2]
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
            points.append(self.globalToImage(x_i, y_i, z)[0])

        points = np.array(points).reshape((-1, 1, 2))

        return points

