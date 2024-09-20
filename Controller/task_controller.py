from enum import Enum
import sys
sys.path.append('D:/Robot 2SR/2sr-swarm-control')
from Model import global_var, robot2sr, splines
from View import plotlib
import motive_client, keyboard_controller, robot2sr_controller, camera_optitrack_synchronizer as cos
import random as rnd
import numpy as np
import pandas as pd
from datetime import datetime
from itertools import chain
from scipy.interpolate import interp1d
from typing import List
import time
import json


class Mode(Enum):
    MANUAL = 1 # Manual control of a single robot via a keyboard
    PATH_TRACKING = 2 # Path tracking
    SOFT = 3

class PathShape(Enum):
    curve = 1
    ellipse = 2 
    lemniscate = 3
    spiral = 4

class Task(keyboard_controller.ActionsHandler):
    def __init__(self, mode: Mode) -> None:
        super().__init__(global_var.OMNI_SPEED, global_var.ROTATION_SPEED, global_var.LU_SPEED)
        self.mode = mode

        self.mocap = motive_client.MocapReader() # Initialise the reader of the tracking data
        self.rgb_camera = cos.Aligner()
        self.gui = plotlib.GUI() # Initialise GUI

        self.markers = {}
        self.agent: robot2sr.Robot = None
        self.agent_controller = robot2sr_controller.Controller()

        self.cardioid1 = splines.Cardioid(1)
 
        self.tracking_area = [[-1, 3], [-1, 3]]

        self.simulation = False
        

    @property
    def mode(self) -> Mode:
        return self.__mode

    @mode.setter
    def mode(self, value) -> None:
        if not isinstance(value, Mode):
            raise Exception('Wrong task mode!')
        self.__mode = value        


    # Execute the task of a given mode
    def run(self) -> None:
        print('Start Motive streaming')
        self.mocap.startDataListener() # Start listening data from Motive
        
        print('Start the experiment')
        self.__updateConfig() # Update MAS and manipulandums
        
        # Plot agents
        # self.gui.plotAgent(self.agent)
        
        # Run the seleceted task mode
        match self.mode:
            case Mode.MANUAL:
                print('Manual mode')
                self.__manualMode()
            case Mode.PATH_TRACKING:
                print('Path tracking mode')
                # self.__pathTrackingMode()
                self.__testSoftStates()
            case Mode.SOFT:
                print('Soft mode')
                self.__softMode()

        self.gui.window.mainloop() # Start the GUI application

    def __updateConfig(self):
        # Get the current MAS and manipulandums configuration
        agent_config, self.markers, msg = self.mocap.getAgentConfig()

        if agent_config:
            if self.agent:
                self.agent.pose = [agent_config['x'], agent_config['y'], agent_config['theta']]
                self.agent.k1 = agent_config['k1']
                self.agent.k2= agent_config['k2']
            else:
                self.agent = robot2sr.Robot(agent_config['id'], agent_config['x'], agent_config['y'], agent_config['theta'], agent_config['k1'], agent_config['k2'])

            self.agent.head.pose = agent_config['head']
            self.agent.tail.pose = agent_config['tail']
        else:
            print('Agent is not detected! ' + msg)


    #//////////////////////////////// MANUAL MODE METHODS ////////////////////////////////
    
    def __manualMode(self):
        # Handle key events 
        self.gui.window.bind('<KeyPress>', self.__onPress)
        self.gui.window.bind('<KeyRelease>', self.__onRelease)

    # Execute the action according to the keyboard commands
    def __executeAction(self):
        self.__updateConfig()
        if self.agent is not None:
            self.agent_controller.move(self.agent, self.v, self.s)
            # Update the GUI
            # self.gui.plotAgent(self.agent, self.markers)

    def __onPress(self, key) -> None:
        super().onPress(key)
        self.__executeAction()

    def __onRelease(self, key) -> None:
        super().onRelease(key)
        self.__executeAction()

    #//////////////////////////// PATH TRACKING MODE METHODS //////////////////////////////
    
    def  __pathTrackingMode(self):

        ps = PathShape.lemniscate
        # self.simulation = True

        while not self.agent: 
            self.__updateConfig()

        home_pose = self.agent.pose

        path = self.__generatePath(ps)
        date_title = ps.name + '_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.rgb_camera.startVideo(path.x, path.y, self.agent.config, date_title)

        print("Waiting for the video to start...")
        while not self.rgb_camera.wait_video:
            pass

        print('Video started')
        print()

        goal = path.getPoint(len(path.x) - 1)
        # states = self.__generateStates(path)
        dist = splines.getDistance(self.agent.position, goal)

        safety_margin = 2
        counter = 0
        v_prev = [0, 0]
        elapsed_time = 0

        exp_data = []
        robot_tracking_data = []

        time_start = time.perf_counter()

        config_prev = self.agent.pose
        time_prev = time_start

        while dist > 0.02 or elapsed_time < 5:
            time_current = time.perf_counter()
            dt = time_current - time_prev
            time_prev = time_current

            pose_diff = np.array(config_prev) - self.agent.pose
            velocity_global = pose_diff / dt
            velocity_body = np.linalg.pinv(self.agent.jacobian_rigid()[:3, :]) @ velocity_global

            v_current = [velocity_body[1], velocity_body[2]]
            config_prev = self.agent.pose

            v, s = self.agent_controller.motionPlannerMPC(self.agent, path, v_current)
            print(v)

            if counter > safety_margin:
                _, q, status = self.agent_controller.move(self.agent, v, s)
                if self.simulation:
                    self.agent_controller.update_agent(self.agent, q)
                else:
                    self.__updateConfig()
                self.rgb_camera.add_config(self.agent.config)   

            dist = splines.getDistance(self.agent.position, goal)
            target_point = path.getPoint(path.last_idx)
            pose_errors = [target_point[0] - self.agent.x, target_point[1] - self.agent.y, path.yaw[path.last_idx] - self.agent.theta]

            experiment_time_current = time.perf_counter()
            elapsed_time = experiment_time_current - time_start

            # print(f'ref yaw: {path.yaw[path.last_idx]}')
            # print(f'cur yaw: {self.agent.theta}')
            # print()

            
            # Collect data in a JSON file

            robot_pose_data ={'x': self.agent.x, 'y': self.agent.y, 'theta': self.agent.theta, 'k1': self.agent.k1, 
                              'k2': self.agent.k2, 'stiff1': self.agent.stiffness[0], 'stiff2': self.agent.stiffness[1]}
            robot_velocity_data = {'v_x': 0, 'v_y': v_current[0], 'omega': v_current[1]}
            robot_pose_errors_data = {'e_x': pose_errors[0], 'e_y': pose_errors[1], 'e_theta': pose_errors[2]}
            robot_vel_errors_data = {'e_v_x': 0, 'e_v_y': v_prev[0] - v_current[0], 'e_omega':  v_prev[1] - v_current[1]}
            errors_data = {'pose_errors': robot_pose_errors_data, 
                           'vel_errors': robot_vel_errors_data}
            
            robot_tracking_data.append({'time': elapsed_time, 
                                        'pose': robot_pose_data, 
                                        'distance_to_target': dist,
                                        'vel': robot_velocity_data,
                                        'errors': errors_data})


            v_prev = v
            counter += 1

            if self.rgb_camera.finish:
                break

        self.agent_controller.stop(self.agent)
        self.rgb_camera.finish = True

        print()
        print(f'Recording time: {elapsed_time} seconds')

        path_data = []
        for x, y, yaw, in zip(path.x, path.y, path.yaw):
            path_data.append({'x': x, 'y': y, 'yaw': yaw})
            
        # Create the data structure
        data_json = {
            "metadata": {
                "description": "Path tracking plus morphology change data",
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "path": path_data,
            "robot_tracking": robot_tracking_data
        }

        # Write the data to a JSON file
        json_path = f'Experiments/Data/tracking_data_{date_title}.json'
        with open(json_path, 'w') as f:
            json.dump(data_json, f, indent=2)

        print()
        print("JSON file 'tracking_morph_data.json' has been created.")

        # self.__go_home(home_pose)

   
    def __generatePath(self, shape: PathShape) -> splines.Trajectory:        
        if shape == PathShape.curve:
            x_original = np.arange(0, 1.7, 0.01)
            y_original = np.array([np.sin(x / 0.18) * x / 15.0 for x in x_original])

            rot_angle = self.agent.theta + np.pi/2
            rot_matrix = np.array([[np.cos(rot_angle), -np.sin(rot_angle)],
                                [np.sin(rot_angle), np.cos(rot_angle)]])
            coords_rotated = rot_matrix @ [x_original, y_original]
            coords_rotated += np.array(self.agent.position).reshape(2, 1)
        if shape == PathShape.ellipse:
            a, b = 0.55, 0.38
            theta = np.linspace(0, 2*np.pi, 100)
            x_original = a * np.cos(theta)
            y_original = b * np.sin(theta)

            rot_angle = self.agent.theta
            rot_matrix = np.array([[np.cos(rot_angle), -np.sin(rot_angle)],
                                [np.sin(rot_angle), np.cos(rot_angle)]])
            coords_rotated = rot_matrix @ [x_original, y_original]
            coords_rotated += np.array([self.agent.x, self.agent.y + a]).reshape(2, 1)

        if shape == PathShape.lemniscate:
            t = np.linspace(0, 2*np.pi, 150)
            a = 0.75  # controls the size of the figure-8
            x_original = a * np.cos(t) / (np.sin(t)**2 + 1)
            y_original = a * np.cos(t) * np.sin(t) / (np.sin(t)**2 + 1)

            rot_angle = self.agent.theta
            rot_matrix = np.array([[np.cos(rot_angle), -np.sin(rot_angle)],
                                [np.sin(rot_angle), np.cos(rot_angle)]])
            coords_rotated = rot_matrix @ [x_original, y_original]
            coords_rotated += np.array([self.agent.x, self.agent.y + a]).reshape(2, 1)

        path_x, path_y = coords_rotated[0,:].tolist(), coords_rotated[1,:].tolist()

        path = splines.Trajectory(path_x, path_y)
       
        return path
    
    def __generateStates(self, path:splines.Trajectory) -> List[list]:
        max_curvature = np.pi / (2 * global_var.L_VSS)

        path_slices = path.divideIntoThirds()
        states_idx = list(path_slices) + [len(path.traj_x)-1]

        states = {}

        for state_idx in states_idx: 
            position = path.getPoint(state_idx)
            orientation = path.getSlopeAngle(state_idx) - np.pi/2
            orientation %= (2 * np.pi)
            k1 = rnd.uniform(-max_curvature, max_curvature)
            k2 = rnd.uniform(-max_curvature, max_curvature)

            state_config = position + [orientation, k1, k2]

            states[str(state_idx)] = state_config

        return states
    
    def __go_home(self, home_pose) -> None:
        varsigma = [0, 0]

        dx = home_pose[0] - self.agent.x
        dy = home_pose[1] - self.agent.y
        direction = np.arctan2(dy, dx)

        target_angle = direction - np.pi/2

        angle_diff = abs(target_angle - self.agent.theta)
        
        while angle_diff > 0.05:
            theta_dot = 1.65 * (target_angle - self.agent.theta) * global_var.DT
            _, q = self.agent_controller.move(self.agent, [0, 0, theta_dot, 0, 0], [0, 0])

            if self.simulation:
                self.agent_controller.update_agent(self.agent, q)
            else:
                self.__updateConfig()

            self.rgb_camera.add_config(self.agent.config)   

            angle_diff = abs(target_angle - self.agent.theta)

        target = np.array(home_pose)
        dist = np.linalg.norm(self.agent.pose - target)

        while dist > 10**(-1):
            v, varsigma = self.agent_controller.inverse_k(self.agent, home_pose + self.agent.curvature)
            _, q = self.agent_controller.move(self.agent, v, varsigma)

            if self.simulation:
                self.agent_controller.update_agent(self.agent, q)
            else:
                self.__updateConfig()

            self.rgb_camera.add_config(self.agent.config)   

            dist = np.linalg.norm(self.agent.pose - target)
            print(dist)

    #//////////////////////////////// SOFT MODE METHODS ////////////////////////////////
    def __get_lu_pos(self, config, lu=1):
        R = np.array([[np.cos(config[2]), -np.sin(config[2])],
                      [np.sin(config[2]), np.cos(config[2])]])
        
        if lu == 1:
            lu_pos = R @ self.cardioid1.pos(config[3])
        else:
            lu_pos = R @ self.cardioid1.pos(config[4], 2)
        
        lu_pos += config[:2]

        return lu_pos
    
    def __close_to_target(self, target: list) -> bool:
        current = np.array(self.agent.pose)
        target_pose = np.array(target[:3])

        dist = np.linalg.norm(current - target_pose)
        # dist = np.sqrt(((self.agent.x - target[0])**2 + 
        #                 (self.agent.y - target[1])**2 + 
        #                 (self.agent.theta - target[2])**2))
        print(f'Distance: {dist}')

        if dist < 0.015:
            return True

        return False

    
    def __softMode(self):
        # self.simulation = True

        # Measure the initial config
        while not self.agent: 
            self.__updateConfig()

        # ------------------------ Define target ------------------------
        v_soft_target = [-0.032, 0.064] # Control soft velocities
        s = [1, 0] # Control stiffness

        # Target config
        config_traj = self._generateSoftTrajectory(v_soft_target, s, 10)
        config_target = config_traj[-1]

        # config_target[0] += 0.2
        # config_target[1] += 0.3

        lu1_target = self.__get_lu_pos(config_target)
        lu2_target = self.__get_lu_pos(config_target, 2)
        # ---------------------------------------------------------------

        # ------------------------ Start a video ------------------------
        date_title = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.rgb_camera.startVideo(config_target, self.agent.config, date_title)

        print("Waiting for the video to start...")
        while not self.rgb_camera.wait_video:
            pass

        print('Video started')
        print()
        # ---------------------------------------------------------------

        elapsed_time = 0
        time_start = time.perf_counter()

        config_prev = self.agent.config
        time_prev = time_start
        v_prev = [0.0] * 3 + v_soft_target

        robot_tracking_data = []

        rigid_mode = False
        finish = False

        # ------------------------ Run experiment -----------------------
        while True:
            # Get target velocities
            if rigid_mode:
                if self.__close_to_target(config_target) or self.rgb_camera.finish:
                    v_rigid = [0.0] * 3
                    finish = True
                else:
                    v_rigid = self.agent_controller.mpcRM(self.agent, config_target, v_prev[:3])

                v_soft = [0.0, 0.0]
                s = [0, 0]
            else:
                if abs(self.agent.k1 - config_target[3]) < 5 or self.rgb_camera.finish:
                    v_soft = [0.0, 0.0]
                    s = [0, 0]
                    rigid_mode = True
                else:
                    v_soft, lu1, lu2 = self.agent_controller.mpcSM1(self.agent, config_target, lu1_target, lu2_target, v_prev[3:])
                    s = [1, 0]

                v_rigid = [0.0] * 3

            # Move robot
            v = v_rigid + v_soft
            print(f'Target velocities: {v}')
            print(f'Target stiffness: {s}')
            print()

            _, q, s_current, sc_feedback = self.agent_controller.move(self.agent, v, s)
            
            # Update robot's state
            if self.simulation:
                self.agent_controller.update_agent(self.agent, q)
                self.agent.stiffness = s
            else:
                self.__updateConfig()
                self.agent.stiffness = s_current
            
            self.rgb_camera.add_config(self.agent.config)   

            # Calculate experiment data
            config_errors = [q[0] - self.agent.x, q[1] - self.agent.y, q[2] - self.agent.theta, 
                             q[3] - self.agent.k1, q[4] - self.agent.k2]
            
            target_errors = [config_target[0] - self.agent.x, config_target[1] - self.agent.y, config_target[2] - self.agent.theta, 
                             config_target[3] - self.agent.k1, config_target[4] - self.agent.k2]
            
            time_current = time.perf_counter()
            dt = time_current - time_prev
            config_diff = np.array(config_prev) - self.agent.config
            velocity_global = config_diff / dt
            velocity_body = np.linalg.pinv(self.agent.jacobian(self.agent.stiffness)) @ velocity_global

            elapsed_time = time_current - time_start

            # Collect data in dictionaries
            robot_config_data ={'x': self.agent.x, 'y': self.agent.y, 'theta': self.agent.theta, 'k1': self.agent.k1, 
                              'k2': self.agent.k2, 'stiff1': self.agent.stiffness[0], 'stiff2': self.agent.stiffness[1]}
            robot_velocity_data = {'v_x': velocity_body[0], 'v_y': velocity_body[1], 'omega': velocity_body[2], 'v_1': velocity_body[3], 'v_2': velocity_body[4]}
            robot_config_errors_data = {'e_x': config_errors[0], 'e_y': config_errors[1], 'e_theta': config_errors[2], 'e_k1': config_errors[3], 'e_k2': config_errors[4]}
            robot_target_errors_data = {'e_x': target_errors[0], 'e_y': target_errors[1], 'e_theta': target_errors[2], 'e_k1': target_errors[3], 'e_k2': target_errors[4]}
            robot_vel_errors_data = {'e_v_x': v[0] - velocity_body[0], 'e_v_y': v[1] - velocity_body[1], 'e_omega': v[2] - velocity_body[2], 
                                     'e_v_1': v[3] - velocity_body[3], 'e_v_2': v[4] - velocity_body[4]}
            errors_data = {'config_errors': robot_config_errors_data, 
                           'target_errors': robot_target_errors_data,
                           'vel_errors': robot_vel_errors_data}
            sc_data = {'temp': sc_feedback[0],
                       'time': sc_feedback[1]}
            
            robot_tracking_data.append({'time': elapsed_time, 
                                        'config': robot_config_data, 
                                        'vel': robot_velocity_data,
                                        'errors': errors_data,
                                        'stiff_trans': sc_data})
            
            # Update prev vars
            config_prev = self.agent.config
            v_prev = v
            time_prev = time_current

            if finish:
                self.rgb_camera.finish = True
                break

        print()
        print(f'Recording time: {elapsed_time} seconds')

        # Create the data structure
        data_json = {
            "metadata": {
                "description": "Reaching a target configuration SMM1",
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "robot_tracking": robot_tracking_data
        }

        # Save data to a JSON file
        json_path = f'Experiments/Data/Tracking/SM1/sm1_data_{date_title}.json'
        with open(json_path, 'w') as f:
            json.dump(data_json, f, indent=2)

        print()
        print("JSON file 'tracking_morph_data.json' has been created.")

        # self.gui.plot_config(config_traj[0], self.agent.stiffness, 'initial')
        # # self.gui.scatter(config_traj_array[0,:], config_traj_array[1,:], 'original traj')
        # self.gui.plot_config(config_traj[-1], self.agent.stiffness, 'target', 'target')
        # self.gui.plot_config(self.agent.config, self.agent.stiffness, 'result')
        # # self.gui.plot(*lu1_target)
        # # self.gui.plot(*lu2_target)
        # # self.gui.plot(*lu1)
        # # self.gui.plot(*lu2)
        # self.gui.show()

    def _generateSoftTrajectory(self, v, s, timer):
        q_start = self.agent.config
        v_all = [0] * 3 + v

        traj = [q_start]

        t = 0

        while t < timer:
            q = self.agent_controller.get_config(self.agent, v_all, s)
            # print(splines.getDistance(self.agent.position, q[:2]))
            self.agent_controller.update_agent(self.agent, q)

            traj.append(q)
            t += 1 

        self.agent.config = q_start

        return traj

    def __testSoftStates(self):
        k_max = np.pi / (2 * global_var.L_VSS)
        max_values = [2.78, 1.4, 2*np.pi, k_max, k_max]

        q_start = [0, 0, 0, 0.001, 0]
        self.agent = robot2sr.Robot(1, *q_start, stiffness=[1, 0])
        self.agent_controller.update_agent(self.agent, self.agent.config)

        v_fk = [0, 0, 0, 0, 0.15]

        print()
        print(f'Original velocity v2: {v_fk[-1]}')
        print()

        original_traj_x = []
        original_traj_y = []
        counter = 0

        self.agent.theta0 = self.agent.theta

        print('Original wheels\' velocities:' )

        while counter < 10:
            _, q = self.agent_controller.move(self.agent, v_fk, self.agent.stiffness)
            self.agent_controller.update_agent(self.agent, q)

            original_traj_x.append(q[0])
            original_traj_y.append(q[1])
            counter += 1

            if np.abs(q[3]) >= k_max or np.abs(q[4]) >= k_max:
                break

        print()

        target_config = self.agent.config
        # print(f'Target: {target_config}')
        self.agent_controller.update_agent(self.agent, q_start)


        #------------------------------------------------------------------------------
        # Generate reference trajectory

        ref_traj = []

        N = 9
        k_array = np.linspace(q_start[3], target_config[3], N+1)
        spiral1 = splines.LogSpiral(1)
        
        for k in k_array:
            ref_pos = spiral1.get_pos(k)

            ref_traj.append(ref_pos)

        rot = np.array([[np.cos(q_start[2]), -np.sin(q_start[2])],
                        [np.sin(q_start[2]), np.cos(q_start[2])]])
        
        ref_traj = rot @ np.array(ref_traj).T
        
        offset = [q_start[0] - ref_traj[0, 0], q_start[1] - ref_traj[1, 0]]
        ref_traj += np.array(offset).reshape(2, 1)

        ref_traj_x = ref_traj[0,:]
        ref_traj_y = ref_traj[1,:]

        ref_traj = splines.Trajectory(ref_traj_x, ref_traj_y)

        #------------------------------------------------------------------------------
        v1_predicted, v2_predicted = self.agent_controller.mhe(ref_traj, k_array)
        # print(v1_predicted)
        print('Estimated velocities v2 along the traj:')
        print(v2_predicted)        

        print()
        print('Output wheels\' velocities:')

        #------------------------------------------------------------------------------
        for v2 in v2_predicted:
            v = [0] * 4 + [v2]

            _, q = self.agent_controller.move(self.agent, v, self.agent.stiffness)
            self.agent_controller.update_agent(self.agent, q)
            self.gui.plot_config(self.agent.config, self.agent.stiffness, '')

        self.agent_controller.move(self.agent, [0] * 5, self.agent.stiffness)

        #------------------------------------------------------------------------------
        # goal = ref_traj.getPoint(len(ref_traj.x) - 1)
        # dist = splines.getDistance(self.agent.position, goal)

        # last_dist = 0

        # diff = self.normalized_difference(self.agent.config, target_config, max_values)

        # config_prev = self.agent.config

        # time_prev = time.perf_counter()
        # last_diff = 0
        # self.gui.plot_config(self.agent.config, self.agent.stiffness)


        # while dist > 0.01:
        # #     time_current = time.perf_counter()
        # #     dt = time_current - time_prev
        # #     time_prev = time_current

        # #     config_diff = np.array(config_prev) - self.agent.config
        # #     q_dot = config_diff / dt
        # #     vel_current = np.linalg.pinv(self.agent.jacobian_soft([1, 0])) @ q_dot
        #     x_ref, y_ref, th_ref, k1_ref, k2_ref = [], [], [], [], []

        #     n = len(ref_traj.x)
        #     nearest_ind = ref_traj.getTarget(self.agent.position, 0.007)
        #     for i in range(0, self.agent_controller.T):
        #         if nearest_ind + i < n:
        #             x_ref.append(ref_traj.x[nearest_ind + i])
        #             y_ref.append(ref_traj.y[nearest_ind + i])
        #             th_ref.append(ref_traj.yaw[nearest_ind + i])
        #             k1_ref.append(k_array[nearest_ind + i])
        #         else:
        #             x_ref.append(ref_traj.x[n-1])
        #             y_ref.append(ref_traj.y[n-1])
        #             th_ref.append(ref_traj.yaw[n-1])
        #             k1_ref.append(k_array[n-1])
                    
        #         k2_ref.append(self.agent.k2)

        #     q_ref = [x_ref, y_ref, th_ref, k1_ref, k2_ref]

        #     v1, v2 = self.agent_controller.motionPlannerMPC10(self.agent, q_ref)
        #     v = [0] * 3 + [v1, v2]
        #     print(f'v1: {v1}, v2: {v2}')

        #     _, q = self.agent_controller.move(self.agent, v, self.agent.stiffness)
        #     self.agent_controller.update_agent(self.agent, q)
        #     self.gui.plot_config(self.agent.config, self.agent.stiffness)

        #     dist = splines.getDistance(self.agent.position, goal)

        # #     diff = self.normalized_difference(self.agent.config, target_config, max_values)
        #     print()
        #     print(f'Distance: {dist}')
        #     print()

        #     if abs(dist - last_dist) == 0:
        #         break

        # #     last_diff = diff
        #     last_dist = dist
        
        # self.gui.clear()
        # self.gui.plot_config(q_start, self.agent.stiffness, '')
        # self.gui.scatter(original_traj_x, original_traj_y, 'original traj')
        # self.gui.scatter(ref_traj_x, ref_traj_y, 'estimated traj')
        # self.gui.plot_config(target_config, self.agent.stiffness, 'target', 'target')
        # self.gui.show()

    
    @staticmethod
    def normalized_difference(q1, q2, max_values):
        # Ensure inputs are numpy arrays
        q1 = np.array(q1)
        q2 = np.array(q2)
        max_values = np.array(max_values)
        
        # Normalize the difference
        normalized_diff = (q1 - q2) / max_values
        
        # Calculate the Euclidean norm of the normalized difference
        norm_diff = np.linalg.norm(normalized_diff)
        
        return norm_diff

    
if __name__ == "__main__":
    experiment = Task(Mode.SOFT)
    experiment.run()
