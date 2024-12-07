import sys
sys.path.append('D:/Robot 2SR/2sr-swarm-control')
from Model import robot2sr as rsr, manipulandum
import motive_client, camera_optitrack_synchronizer as cos
import threading
from datetime import datetime
import numpy as np
import cv2
from shapely.geometry import Polygon

class Observer:
    def __init__(self, simulation=False) -> None:
        self.simulation = simulation

        self.markers = None
        self.agent: rsr.Robot = None
        self.object: manipulandum.Shape = None
        self.object_target: manipulandum.Shape = None

        self.original_obstacles = []
        self.expanded_obstacles = []

        self.mocap = motive_client.MocapReader()
        self.rgb_camera = cos.Aligner()

        self.date_title = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    def __updateConfig(self) -> None:
        agents_config, objects_config, self.markers, msg = self.mocap.getConfig()

        if agents_config:
            agent_config = agents_config[0]
            if self.agent is not None:
                if not self.simulation:
                    self.agent.pose = [agent_config['x'], agent_config['y'], agent_config['theta']]
                    self.agent.k1 = agent_config['k1']
                    self.agent.k2= agent_config['k2']
                    self.agent.head.pose = agent_config['head']
                    self.agent.tail.pose = agent_config['tail']
            else:
                self.agent = rsr.Robot(agent_config['id'], agent_config['x'], agent_config['y'], agent_config['theta'], agent_config['k1'], agent_config['k2'])
        else:
            print(msg)

        if objects_config:
            object_config = objects_config[0]
            object_pose = [object_config['x'], object_config['y'], object_config['theta']]
            if self.object is not None:
                if not self.simulation:
                    self.object.pose = object_pose
            else:
                self.object = manipulandum.Shape(object_config['id'], object_pose)

    def __updateConfigLoop(self) -> None:
        while True:
            self.__updateConfig()

            self.rgb_camera.markers = self.markers
            if self.object is not None:
                self.rgb_camera.manip_center = self.object.pose

    def run(self) -> None:
        print('Start Motive streaming...\n')
        self.mocap.startDataListener()

        update_thread = threading.Thread(target=self.__updateConfigLoop)
        update_thread.daemon = True  
        update_thread.start()

        self.rgb_camera.startVideo(self.date_title)

        print('\nWaiting for the video to start...\n')
        while not self.rgb_camera.wait_video:
            pass

        print('Video started\n')

    def detectObstacles(self) -> None:
        self.rgb_camera.detect_status = True

        while True:
            if self.rgb_camera.obstacles is not None:
                break

        obstacles_corners = self.rgb_camera.obstacles
    
        contour = self.object.contour.T
        # Calculate perimeter
        perimeter = cv2.arcLength(contour.astype(np.float32), closed=True)
        
        # Approximate polygon
        epsilon = 0.01 * perimeter
        approx = cv2.approxPolyDP(contour.astype(np.float32), epsilon, closed=True)

        obstacles_corners.append(approx.squeeze())

        for corners in obstacles_corners:

            obstacle_poly = Polygon(corners)
            self.original_obstacles.append(obstacle_poly)
            
            worst_case_radius = 0.02
            expanded_poly = obstacle_poly.buffer(worst_case_radius, join_style=2)
            self.expanded_obstacles.append(expanded_poly)

            expanded_corners = []
            xx, yy = expanded_poly.exterior.coords.xy
            for x, y in zip(xx.tolist()[:-1], yy.tolist()[:-1]):
                expanded_corners.append((x, y))
            self.rgb_camera.expanded_obstacles_global.append(expanded_corners)

    def defineTargetObject(self, target_pose) -> None:
        self.object_target = manipulandum.Shape(self.object.id, target_pose)
        self.rgb_camera.manip_target_contour = self.object_target.contour

    def showObjPath(self, path) -> None:
        self.rgb_camera.path = path
