import json
import numpy as np
import cv2
import threading
import time
from scipy.interpolate import splprep, splev
from enum import Enum
from Model import global_var as gv

neon_red = (51, 87, 255)
neon_blue = (255, 217, 4)
neon_green = (20, 255, 57)
neon_yellow = (31, 240, 255)
yellow = (28, 188, 244)
neon_orange = (8, 178, 249)

class Shape(Enum):
    CIRCLE = 1
    ELLIPSE = 2
    OTHER = 3

class Aligner:
    def __init__(self) -> None:
        self.file_path = "Controller/calibration_data.json"
        self.data = None

        self.wait_video = False
        self.finish = False

        self.current_shape = Shape.ELLIPSE
        self.detected_object_contour = None
        
        self.markers = None

        self.contour = None
        self.manip_center = None

        self.manip_target_contour = None
        self.traversed_trajectory = []

        self.path = None

        self.contact_points = []
        self.c_dot = []

        #--------------------------------------------------
        self.obstacles_contour = []
        self.obstacles = None
        self.expanded_obstacles_global = []
        self.expanded_obstacles = None

        self.direction = None
        self.grasp_point = None

        self.rrt_path = None
        self.heading = None
        #--------------------------------------------------
        self.new_pos = None
        self.all_nodes = []

        #--------------------------------------------------

        try:
            with open(self.file_path, "r") as json_file:
                data = json.load(json_file)
                self.data = data
        except FileNotFoundError:
            print(f"Error: {self.file_path} not found.")
            return

        self.__read_camera_calibration_data()

    def add2traj(self, pose, heading=None) -> None:
        self.traversed_trajectory.append(pose)
        self.heading = heading

    def startVideo(self, date_title: str, task: str, args=[]):
        if task == 'soft_modes':
            config_target, config0 = args
            self.config_list.append(config0)
            thread = threading.Thread(target=self.__run_old, args=(config_target, date_title))
            thread.start()
        elif task == 'object_handling':
            thread = threading.Thread(target=self.__run_tr, args=(date_title,))
            thread.start()
        elif task == 'object_grasp':
            thread = threading.Thread(target=self.__run_gr, args=(date_title,))
            thread.start()

    def __run_tr(self, date_title: str):
        video_path_rgb = f'Experiments/Video/Grasping/transport_bean_{date_title}.mp4'

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path_rgb, fourcc, 16.0, (1080,520))

        cap_rgb = cv2.VideoCapture(0)
        # set the resolution to 1280x720
        cap_rgb.set(3, 1280)
        cap_rgb.set(4, 720)

        start_timer = False
        time_start = time.perf_counter()
        elapsed_time = 0

        while cap_rgb.isOpened():
            _, frame = cap_rgb.read()
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            h, w = frame.shape[:2]
            self.new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist, (w,h), 1, (w,h))
            undistorted_frame = cv2.undistort(frame, self.camera_matrix, self.dist, None, self.new_camera_matrix)

            mean_z = 0

            if self.markers is not None:
                z_values = [marker['marker_z'] for marker in self.markers.values()]
                if len(z_values) != 0:
                    mean_z = sum(z_values) / len(z_values)

                depth_list = []

                for marker in self.markers.values():
                    _, depth = self.globalToImage(marker['marker_x'], marker['marker_y'], marker['marker_z'])
                    depth_list.append(depth)

                # depth = sum(depth_list) / len(depth_list)
                # self.detect_shape(undistorted_frame, depth)
                
            
            if self.path is not None:
                points = []
                for p in self.path:
                    points.append(self.globalToImage(*p, mean_z)[0])
                
                path = np.array(points).reshape((-1, 1, 2))
                cv2.polylines(undistorted_frame, [path], False, neon_blue, 1)
            
            contour_points = []
            if self.contour is not None:
                for i in range(self.contour.shape[1]):
                    p, _ = self.globalToImage(*self.contour[:,i], mean_z)
                    contour_points.append(p)
            contour_array = np.array(contour_points).reshape((-1, 1, 2))
            cv2.polylines(undistorted_frame, [contour_array], True, neon_blue, 2)

            tracked_points = []
            for p in self.traversed_trajectory:
                pos, _ = self.globalToImage(*p, mean_z)
                tracked_points.append(pos)
                # cv2.circle(undistorted_frame, pos, 2, neon_red, -1)

            tracked_traj = np.array(tracked_points).reshape((-1, 1, 2))
            cv2.polylines(undistorted_frame, [tracked_traj], False, neon_green, 2)

            if self.manip_center is not None:
                (x, y), _ = self.globalToImage(*self.manip_center[:-1], mean_z)
                cv2.circle(undistorted_frame, (x, y), 4, (0, 0, 0), -1)

            if self.manip_center is not None:
                (x, y), _ = self.globalToImage(*self.manip_center[:-1], mean_z)
                cv2.circle(undistorted_frame, (x, y), 4, (0, 0, 0), -1)

            # if len(self.contact_points) == len(self.c_dot):
            #     for cp, c_dot_i in zip(self.contact_points, self.c_dot):
            #         (x, y), _ = self.globalToImage(*cp[:-1], mean_z)
            #         cv2.circle(undistorted_frame, (x, y), 3, (255, 0, 255), -1)

            #         dt = 3
            #         end_x_global = cp[0] + c_dot_i[0] * dt
            #         end_y_global = cp[1] + c_dot_i[1] * dt

            #         (end_x, end_y), _ = self.globalToImage(end_x_global, end_y_global, mean_z)

            #         # Draw the vector
            #         cv2.arrowedLine(undistorted_frame, (x, y), (end_x, end_y), (255, 0, 255), 2)

            target_contour_points = []
            if self.manip_target_contour is not None:
                for i in range(self.manip_target_contour.shape[1]):
                    p, _ = self.globalToImage(*self.manip_target_contour[:,i], mean_z)
                    target_contour_points.append(p)
            target_contour_array = np.array(target_contour_points).reshape((-1, 1, 2))
            cv2.polylines(undistorted_frame, [target_contour_array], True, neon_blue, 2)
                

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
    
    def __run_gr(self, date_title: str):
        video_path_rgb = f'Experiments/Video/Grasping/grasp_heart_{date_title}.mp4'

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path_rgb, fourcc, 16.0, (1080,520))

        cap_rgb = cv2.VideoCapture(0)
        # set the resolution to 1280x720
        cap_rgb.set(3, 1280)
        cap_rgb.set(4, 720)

        start_timer = False
        time_start = time.perf_counter()
        elapsed_time = 0

        while cap_rgb.isOpened():
            _, frame = cap_rgb.read()
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            h, w = frame.shape[:2]
            self.new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist, (w,h), 1, (w,h))
            undistorted_frame = cv2.undistort(frame, self.camera_matrix, self.dist, None, self.new_camera_matrix)

            mean_z = 0
            depth = 0

            if self.markers is not None:
                z_values = [marker['marker_z'] for marker in self.markers.values()]
                if len(z_values) != 0:
                    mean_z = sum(z_values) / len(z_values)

                depth_list = []

                for marker in self.markers.values():
                    _, depth = self.globalToImage(marker['marker_x'], marker['marker_y'], marker['marker_z'])
                    depth_list.append(depth)

                depth = sum(depth_list) / len(depth_list)
                
            
            if self.obstacles is None:
                self.detect_obstacles(undistorted_frame, depth)

            # for obstacle_contour in self.obstacles_contour:
            #     cv2.polylines(undistorted_frame, [obstacle_contour], True, neon_green, 1)

            if self.expanded_obstacles is None:
                if len(self.expanded_obstacles_global) > 0:
                    self.expanded_obstacles = []
                    for expanded_obstacle_global in self.expanded_obstacles_global:
                        expanded_obstacle = []
                        for x, y in expanded_obstacle_global:
                            x_, y_ = self.globalToImage(x, y, mean_z)[0]
                            expanded_obstacle.append([[x_, y_]])
                        self.expanded_obstacles.append(np.array(expanded_obstacle))
            else:
                for obstacle_contour in self.expanded_obstacles:
                    cv2.polylines(undistorted_frame, [obstacle_contour], True, neon_blue, 1)

            if self.direction is not None:
                (x, y), _ = self.globalToImage(*self.manip_center[:-1], mean_z)

                end_x_global = self.manip_center[0] + np.cos(self.direction) * 0.15
                end_y_global = self.manip_center[1] + np.sin(self.direction) * 0.15

                (end_x, end_y), _ = self.globalToImage(end_x_global, end_y_global, mean_z)

                # Draw the vector
                cv2.arrowedLine(undistorted_frame, (x, y), (end_x, end_y), neon_blue, 2)
                cv2.circle(undistorted_frame, (x, y), 4, (0, 0, 0), -1)    

            if self.grasp_point is not None:
                (x, y), _ = self.globalToImage(*self.grasp_point, mean_z)
                cv2.circle(undistorted_frame, (x, y), 3, neon_blue, -1)

            #-----------------------------------------------------------------------------------
            if self.new_pos is not None:
                (x, y), _ = self.globalToImage(*self.new_pos, mean_z)
                cv2.circle(undistorted_frame, (x, y), 3, (0, 0, 255), -1)    

            for node in self.all_nodes:
                (x, y), _ = self.globalToImage(node.position[0], node.position[1], mean_z)
                cv2.circle(undistorted_frame, (x, y), 3, (255, 255, 255), -1)  

                end_x_global = node.position[0] + np.cos(node.theta) * 0.05
                end_y_global = node.position[1] + np.sin(node.theta) * 0.05

                (end_x, end_y), _ = self.globalToImage(end_x_global, end_y_global, mean_z)
                cv2.arrowedLine(undistorted_frame, (x, y), (end_x, end_y), neon_red, 2)

                if node.parent is not None:
                    (x0, y0), _ = self.globalToImage(node.parent.position[0], node.parent.position[1], mean_z)
                    cv2.line(undistorted_frame, (int(x0), int(y0)), (int(x), int(y)), (255, 255, 255), 1)


            #-----------------------------------------------------------------------------------

            if self.rrt_path is not None:
                points = []
                arrow_points = []
                for p in self.rrt_path:
                    points.append(self.globalToImage(*p[:-1], mean_z)[0])
                    # points.append(self.globalToImage(*p, mean_z)[0])

                    # end_x_global = p[0] + np.cos(p[2]) * 0.1
                    # end_y_global = p[1] + np.sin(p[2]) * 0.1

                    # arrow_points.append(self.globalToImage(end_x_global, end_y_global, mean_z)[0])
                
                path = np.array(points).reshape((-1, 1, 2))
                cv2.polylines(undistorted_frame, [path], False, neon_green, 1)

                # for p, ap in zip(points, arrow_points):
                #     cv2.circle(undistorted_frame, p, 3, (0, 255, 0), -1)
                #     cv2.arrowedLine(undistorted_frame, p, ap, neon_red, 2)

            tracked_points = []
            for p in self.traversed_trajectory:
                pos, _ = self.globalToImage(*p, mean_z)
                tracked_points.append(pos)

            tracked_traj = np.array(tracked_points).reshape((-1, 1, 2))
            cv2.polylines(undistorted_frame, [tracked_traj], False, (0, 0, 255), 2)

            # if self.heading is not None:
            #     p_start_global = self.traversed_trajectory[-1]
            #     p_start, _ = self.globalToImage(*p_start_global, mean_z)

            #     end_x_global = p_start_global[0] + np.cos(self.heading) * 0.1
            #     end_y_global = p_start_global[1] + np.sin(self.heading) * 0.1

            #     p_end, _ = self.globalToImage(end_x_global, end_y_global, mean_z)

            #     cv2.arrowedLine(undistorted_frame, p_start, p_end, (0, 0, 255), 2)

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
    
    
    def detect_shape(self, frame, depth):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)

        edges = cv2.Canny(blurred, 50, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            largest_contour = largest_contour.reshape(-1, 1, 2)

            curve_points = []

            if self.current_shape == Shape.CIRCLE and len(largest_contour) >= 5:
                # Fit a circle to the largest contour
                center, radius = cv2.minEnclosingCircle(largest_contour)
                # Draw the circle on the frame
                cv2.circle(frame, center, int(radius), (0, 255, 0), 2)  

                theta = np.linspace(0, 2 * np.pi, 50)
                circle_points = np.array([center[0] + radius * np.cos(theta), 
                                          center[1] + radius * np.sin(theta)]).T
                curve_points.extend(circle_points.reshape((-1, 1, 2)).astype(np.int32))

            if self.current_shape == Shape.ELLIPSE and len(largest_contour) >= 5:  
                ellipse = cv2.fitEllipse(largest_contour)
                cv2.ellipse(frame, ellipse, (0, 255, 0), 2)  

                center = ellipse[0] 
                a = ellipse[1][0] / 2
                b = ellipse[1][1] / 2
                angle = np.radians(ellipse[2])

                theta = np.linspace(0, 2 * np.pi, 50)
                ellipse_points = np.array([center[0] + a * np.cos(theta) * np.cos(angle) - b * np.sin(theta) * np.sin(angle),
                                           center[1] + a * np.cos(theta) * np.sin(angle) + b * np.sin(theta) * np.cos(angle)]).T
                curve_points = np.round(ellipse_points.reshape((-1, 1, 2))).astype(np.int32)

            if self.current_shape == Shape.OTHER:
                # Draw the contour on the frame
                cv2.drawContours(frame, [largest_contour], 0, (0, 255, 0), 2)

                # Create a smooth curve from the largest contour using spline interpolation
                contour_points = largest_contour[:, 0, :]  # Extract points from the contour
                tck, u = splprep(contour_points.T, s=48)  # Spline representation
                smooth_points = np.array(splev(np.linspace(0, 1, 110), tck)).T  # Evaluate spline

                # Draw the smooth curve on the frame
                curve_points = smooth_points.reshape((-1, 1, 2)).astype(np.int32)
                cv2.polylines(frame, [curve_points], isClosed=True, color=(0, 255, 0), thickness=2)

            self.detected_object_contour = []
            
            for point in curve_points:
                x, y = point[0][0], point[0][1]
                self.detected_object_contour.append(self.imageToGlobal((x, y), depth))

            window = 5  # Define the window size
            if len(self.detected_object_contour) >= window:
                self.detected_object_contour = [
                    np.mean(self.detected_object_contour[i - window // 2:i + window // 2 + 1], axis=0).tolist()
                    for i in range(window // 2, len(self.detected_object_contour) - window // 2)
                ]

    def detect_obstacles(self, frame, depth):
        # Convert the frame to the HSV color space
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define the range for blue color in HSV, including lighter shades
        lower_blue = np.array([90, 100, 0])  # Adjusted lower bound to capture lighter shades
        upper_blue = np.array([140, 255, 255])

        # Create a mask for blue color
        blue_mask = cv2.inRange(hsv_frame, lower_blue, upper_blue)

        # Find contours in the mask
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        obstacles_contour = []

        for contour in contours:
            # Approximate the contour to a polygon
            epsilon = 0.04 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Check if the approximated contour has four points (rectangle) and is large enough
            if len(approx) == 4 and cv2.contourArea(approx) > 1000:  # Adjust the area threshold as needed
                obstacles_contour.append(approx)
        
        if len(obstacles_contour) == 4:
            if self.obstacles is None:
                self.obstacles = []

            self.obstacles_contour = obstacles_contour

            for contour in obstacles_contour:
                obstacle = []
                for point in contour:
                    point_global = self.imageToGlobal(point[0], depth)
                    obstacle.append(point_global[:-1].tolist())
                self.obstacles.append(np.array(obstacle))

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

