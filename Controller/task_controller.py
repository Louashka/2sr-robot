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
 
        self.tracking_area = [[-1, 3], [-1, 3]]
        

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
                self.__pathTrackingMode()

        # self.gui.window.mainloop() # Start the GUI application

    def start(self) -> None:
        pass

    def stop(self) -> None:
        self.mas.stop()

    def quit(self) -> None:
        pass

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

        while not self.agent: 
            self.__updateConfig()

        home_pose = self.agent.pose

        path = self.__generatePath()
        self.rgb_camera.startVideo(path.x, path.y, self.agent.config)

        print("Waiting for the video to start...")
        while not self.rgb_camera.wait_video:
            pass

        print('Video started')

        goal = path.getPoint(len(path.x) - 1)
        # states = self.__generateStates(path)
        dist = splines.getDistance(self.agent.position, goal)

        safety_margin = 2
        counter = 0

        exp_data = []
        robot_tracking_data = []

        experiment_start_time = time.perf_counter()

        config_last = self.agent.pose
        last_time = experiment_start_time
        elapsed_time = 0

        while dist > 10**(-2):
            
        #     # v, s = self.agent_controller.motionPlanner(self.agent, path, states)

            current_time = time.perf_counter()
            dt = current_time - last_time
            last_time = current_time

            pose_diff = np.array(config_last) - self.agent.pose
            velocity_global = pose_diff / dt
            velocity_body = self.agent.jacobian_rigid([0, 0])[:3, :] @ velocity_global

            v_current = [velocity_body[1], velocity_body[2]]
            # v_current = [v[1], v[2]]
            config_last = self.agent.pose

            v, s = self.agent_controller.motionPlannerMPC(self.agent, path, v_current)
        #     # v = [0, 0.1, 0, 0, 0]
        #     # print(v)
            if counter > safety_margin:
                wheels, q = self.agent_controller.move(self.agent, v, s)
                # self.agent_controller.update_agent(self.agent, q)
                self.__updateConfig()
                self.rgb_camera.add_config(self.agent.config)   

            dist = splines.getDistance(self.agent.position, goal)

            experiment_current_time = time.perf_counter()
            elapsed_time = experiment_current_time - experiment_start_time

            robot_tracking_data.append({'time': elapsed_time, 'x': self.agent.x, 'y': self.agent.y, 'theta': self.agent.theta, 'k1': self.agent.k1, 
                                        'k2': self.agent.k2, 'stiff1': self.agent.stiffness[0], 'stiff2': self.agent.stiffness[1]})

            # print(dist)
            
            # data_row = (self.agent.config.tolist() + self.agent.head.pose + self.agent.tail.pose + 
            #         list(chain(*wheels)) + [elapsed_time])
            # exp_data.append(data_row)  

            counter += 1

            if self.rgb_camera.finish:
                break

        self.agent_controller.stop(self.agent)
        self.rgb_camera.finish = True

        print(f'Recording time: {elapsed_time} seconds')

        path_data = []
        for x, y, yaw, in zip(path.x, path.y, path.yaw):
            path_data.append({'x': x, 'y': y, 'yaw': yaw})
            
        # Create the data structure
        data_json = {
            "metadata": {
                "description": "Path tracking data",
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "path": path_data,
            "robot_tracking": robot_tracking_data
        }

        # Write the data to a JSON file
        with open('Experiments/Data/tracking_data.json', 'w') as f:
            json.dump(data_json, f, indent=2)

        print("JSON file 'tracking_data.json' has been created.")

        self.__go_home(home_pose)

        # column_names = ["x", "y", "angle", "k1", "k2",  
        #                'x_head', 'y_head', 'theta_head', 
        #                'x_tail', 'y_tail', 'theta_tail',
        #             #    'w1_x', 'w1_y', 'w1_theta',
        #             #    'w2_x', 'w2_y', 'w2_theta',
        #             #    'w3_x', 'w3_y', 'w3_theta',
        #             #    'w4_x', 'w4_y', 'w4_theta',
        #                'w1_x', 'w1_y',
        #                'w2_x', 'w2_y',
        #                'w3_x', 'w3_y',
        #                'w4_x', 'w4_y',
        #                "time"]
        # df = pd.DataFrame(exp_data, columns=column_names)
        # df['path_x'] = pd.Series(path.traj_x)
        # df['path_y'] = pd.Series(path.traj_y)

        # state_ind = 1
        # for state in states.values():
        #     df['state{}'.format(state_ind)] = pd.Series(state)
        #     state_ind += 1

        # print("Save experiment data")
        # df.to_csv('Experiments/Data/reach_target.csv', index=False)
   
    def __generatePath(self) -> splines.Trajectory:
        x_original = np.arange(0, 2, 0.01)
        y_original = np.array([np.sin(x / 0.21) * x / 15.0 for x in x_original])

        rot_angle = self.agent.theta + np.pi/2
        rot_matrix = np.array([[np.cos(rot_angle), -np.sin(rot_angle)],
                               [np.sin(rot_angle), np.cos(rot_angle)]])
        coords_rotated = rot_matrix @ [x_original, y_original]

        coords_rotated += np.array(self.agent.position).reshape(2, 1)

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

        target_config = home_pose + self.agent.pose
        target = np.array(home_pose)
        dist = np.linalg.norm(self.agent.pose - home_pose)

        while dist > 10**(-2):
            v, varsigma = self.agent_controller.inverse_k(self.agent, target_config)
            _, q = self.agent_controller.move(self.agent, v, varsigma)
            self.agent_controller.update_agent(self.agent, q)

            self.__updateConfig()
            dist = np.linalg.norm(self.agent.pose - target)

    
if __name__ == "__main__":
    experiment = Task(Mode.PATH_TRACKING)
    experiment.run()
