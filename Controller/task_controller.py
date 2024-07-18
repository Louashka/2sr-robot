from enum import Enum
import sys
sys.path.append('D:/Robot 2SR/2sr-swarm-control')
from Model import global_var, robot2sr, splines
from View import plotlib
import motive_client, keyboard_controller, grasping_controller, robot2sr_controller
import random as rnd
import numpy as np
import pandas as pd
from datetime import datetime
from itertools import chain
from scipy.interpolate import interp1d
from typing import List


class PI:
    def __init__(self, kp=1.0, ki=0.1):
        """
        Define a PID controller class
        :param kp: float, kp coeff
        :param ki: float, ki coeff
        :param kd: float, kd coeff
        """
        self.kp = kp
        self.ki = ki
        self.Pterm = 0.0
        self.Iterm = 0.0
        self.last_error = 0.0

    def control(self, error):
        """
        PID main function, given an input, this function will output a control unit
        :param error: float, error term
        :return: float, output control
        """
        self.Pterm = self.kp * error
        self.Iterm += error * global_var.DT

        self.last_error = error
        output = self.Pterm + self.ki * self.Iterm
        return output


# There are 4 task modes
class Mode(Enum):
    MANUAL = 1 # Manual control of a single robot via a keyboard
    PATH_TRACKING = 2 # Path tracking

class Task(keyboard_controller.ActionsHandler):
    def __init__(self, mode: Mode) -> None:
        super().__init__(global_var.OMNI_SPEED, global_var.ROTATION_SPEED, global_var.LU_SPEED)
        self.mode = mode

        self.mocap = motive_client.MocapReader() # Initialise the reader of the tracking data
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

        self.gui.window.mainloop() # Start the GUI application

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
            self.gui.plotAgent(self.agent, self.markers)

    def __onPress(self, key) -> None:
        super().onPress(key)
        self.__executeAction()

    def __onRelease(self, key) -> None:
        super().onRelease(key)
        self.__executeAction()

    #//////////////////////////// PATH TRACKING MODE METHODS //////////////////////////////
    
    def  __pathTrackingMode(self):
        exp_data = []

        while not self.agent: 
            self.__updateConfig()
        
        # target = np.array([self.agent.x + rnd.uniform(-1, 1), self.agent.y + rnd.uniform(-1, 1), self.agent.theta + rnd.uniform(-np.pi, np.pi)] + self.agent.curvature)
        
        # num_points = 50
        # path_x = np.linspace(self.agent.x, target[0], num_points)
        # path_y = np.linspace(self.agent.y, target[1], num_points)

        path = self.__generatePath()
        goal = path.getPoint(len(path.traj_x) - 1)
        states = self.__generateStates(path)

        dist = splines.getDistance(self.agent.position, goal)
        
        frames = 1000
        counter = 0

        # while counter < frames:
        while dist > 10**(-2):
            
            v, s = self.agent_controller.motionPlanner(self.agent, path)
            # v = [0, 0.1, 0, 0, 0]
            # print(v)
            wheels, q = self.agent_controller.move(self.agent, v, s)
            self.agent.config = q
 
            # self.__updateConfig()

            vss1 = self.gui.arc(self.agent)
            vss1_conn_x = [self.agent.x + vss1[0][-1] - global_var.L_CONN * np.cos(vss1[2]), self.agent.x + vss1[0][-1]]
            vss1_conn_y = [self.agent.y + vss1[1][-1] - global_var.L_CONN * np.sin(vss1[2]), self.agent.y + vss1[1][-1]]

            lu_head_x = vss1_conn_x[0] + np.sqrt(2) / 2 * global_var.LU_SIDE * np.cos(vss1[2] + np.pi + np.pi / 4)
            lu_head_y = vss1_conn_y[0] + np.sqrt(2) / 2 * global_var.LU_SIDE * np.sin(vss1[2] + np.pi + np.pi / 4)
 
            self.agent.head.pose = [lu_head_x, lu_head_y, vss1[2]]

     
            vss2 = self.gui.arc(self.agent, 2)
            vss2_conn_x = [self.agent.x + vss2[0][-1], self.agent.x + vss2[0][-1] + global_var.L_CONN * np.cos(vss2[2])]
            vss2_conn_y = [self.agent.y + vss2[1][-1], self.agent.y + vss2[1][-1] + global_var.L_CONN * np.sin(vss2[2])]

            lu_tail_x = vss2_conn_x[1] + np.sqrt(2) / 2 * global_var.LU_SIDE * np.cos(vss2[2] - np.pi / 4)
            lu_tail_y = vss2_conn_y[1] + np.sqrt(2) / 2 * global_var.LU_SIDE * np.sin(vss2[2] - np.pi / 4)

            self.agent.tail.pose = [lu_tail_x, lu_tail_y, vss2[2]]   

            dist = splines.getDistance(self.agent.position, goal)

            # print(dist)
            
            timeStamp = datetime.now().strftime("%H:%M:%S")
            data = (self.agent.config.tolist() + self.agent.head.pose + self.agent.tail.pose + 
                    list(chain(*wheels)) + [timeStamp])
            exp_data.append(data)  

            counter += 1

        self.agent_controller.stop(self.agent)

        column_names = ["x", "y", "angle", "k1", "k2",  
                       'x_head', 'y_head', 'theta_head', 
                       'x_tail', 'y_tail', 'theta_tail',
                    #    'w1_x', 'w1_y', 'w1_theta',
                    #    'w2_x', 'w2_y', 'w2_theta',
                    #    'w3_x', 'w3_y', 'w3_theta',
                    #    'w4_x', 'w4_y', 'w4_theta',
                       'w1_x', 'w1_y',
                       'w2_x', 'w2_y',
                       'w3_x', 'w3_y',
                       'w4_x', 'w4_y',
                       "time"]
        df = pd.DataFrame(exp_data, columns=column_names)
        df['path_x'] = pd.Series(path.traj_x)
        df['path_y'] = pd.Series(path.traj_y)

        state_ind = 1
        for state in states:
            df['state{}'.format(state_ind)] = pd.Series(state)
            state_ind += 1

        print("Save experiment data")
        df.to_csv('Experiments/Data/reach_target.csv', index=False)
   
    def __generatePath(self) -> splines.Trajectory:
        path_x = np.arange(0, 2, 0.01)
        path_y = np.array([np.sin(x / 0.21) * x / 2.7 for x in path_x]) + self.agent.y
        
        path_x += self.agent.x

        path = splines.Trajectory(path_x, path_y)
       
        return path
    
    def __generateStates(self, path:splines.Trajectory) -> List[list]:
        max_curvature = np.pi / (2 * global_var.L_VSS)

        path_slices = path.divideIntoThirds()
        states_idx = list(path_slices) + [len(path.traj_x)-1]

        states = []

        for state_idx in states_idx: 
            position = path.getPoint(state_idx)
            orientation = path.getSlopeAngle(state_idx) - np.pi/2
            k1 = rnd.uniform(-max_curvature, max_curvature)
            k2 = rnd.uniform(-max_curvature, max_curvature)

            state_config = position + [orientation, k1, k2]

            states.append(state_config)

        return states

    
if __name__ == "__main__":
    experiment = Task(Mode.PATH_TRACKING)
    experiment.run()
