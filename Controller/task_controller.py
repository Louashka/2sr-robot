from enum import Enum
import sys
sys.path.append('D:/Robot 2SR/2sr-swarm-control')
from Model import global_var, robot2sr
from View import plotlib
import motive_client, keyboard_controller, grasping_controller, robot2sr_controller
import random as rnd
import numpy as np
import pandas as pd
from datetime import datetime
from itertools import chain

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

        # while not self.agent: 
        #     self.__updateConfig()

        self.agent = robot2sr.Robot(1, -0.4350321026162654,-0.185221286686026,5.720241850630597,-3.44,-3.327)
        
        target = np.array([self.agent.x + 0.3, self.agent.y + 0.5, self.agent.theta + np.pi/2] + self.agent.curvature)
        dist = np.sqrt((self.agent.x  - target[0])**2 + (self.agent.y  - target[1])**2 + 
                        self.agent_controller.min_angle_distance(self.agent.theta, target[2])**2)
        
        frames = 5000
        counter = 0

        t = global_var.DT

        # while counter < frames:
        while dist > 10**(-1):
            
            v, s = self.agent_controller.motionPlanner(self.agent, target)
            # wheels = self.agent_controller.move(self.agent, v, s)
 
            # self.__updateConfig()
            # print(self.agent.config.tolist())

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

            w1_0 = 2 * np.array([[-0.0275], [0]])
            w2_0 = 2 * np.array([[0.0105], [-0.0275]])
            w3_0 = 2 * np.array([[0.0275], [0]])
            w4_0 = 2 * np.array([[-0.0105], [-0.027]])

            R = np.array([[np.cos(self.agent.head.theta), -np.sin(self.agent.head.theta)],
                    [np.sin(self.agent.head.theta), np.cos(self.agent.head.theta)]]) 
            w1 = np.matmul(R, w1_0).T[0] + self.agent.head.position
            w2 = np.matmul(R, w2_0).T[0] + self.agent.head.position

            R = np.array([[np.cos(self.agent.tail.theta), -np.sin(self.agent.tail.theta)],
                    [np.sin(self.agent.tail.theta), np.cos(self.agent.tail.theta)]])
            w3 = np.matmul(R, w3_0).T[0] + self.agent.tail.position
            w4 = np.matmul(R, w4_0).T[0] + self.agent.tail.position

            wheels = [w1, w2, w3, w4]

            dist = np.sqrt((self.agent.x  - target[0])**2 + (self.agent.y  - target[1])**2 + 
                           self.agent_controller.min_angle_distance(self.agent.theta, target[2])**2)

            # print(dist)
            # error = np.linalg.norm(config[0] - target)
            
            timeStamp = datetime.now().strftime("%H:%M:%S")

            data = (target.tolist() + self.agent.config.tolist() + self.agent.head.pose + self.agent.tail.pose + 
                    list(chain(*wheels)) + [timeStamp])
            exp_data.append(data)  

            counter += 1

            flag_soft = int(s[0] or s[1])
            flag_rigid = int(not (s[0] or s[1]))

            V_ = np.zeros((4, 5))
            for i in range(4):
                w = wheels[i]
                tau = w[0] * np.sin(w[-1]) - w[1] * np.cos(w[-1])
                V_[i, :] = [flag_soft * int(i == 0 or i == 1), -flag_soft * int(i == 2 or i == 3), 
                            flag_rigid * np.cos(w[-1]), flag_rigid * np.sin(w[-1]), flag_rigid * tau]

            V = 1 / global_var.WHEEL_R * V_
            omega = np.matmul(V, v).round(3)
            print(omega)
            
            v_new = np.matmul(np.linalg.pinv(V), omega)
            # print(v_new)

            q_dot = np.matmul(self.agent.jacobian, v_new)
            q = self.agent.config + q_dot * t
            t += global_var.DT

            q[0] += rnd.uniform(-0.0015, 0.0015)
            q[1] += rnd.uniform(-0.0015, 0.0015)
            q[2] += rnd.uniform(-0.01, 0.01)
            q[3] += rnd.uniform(-0.2, 0.2)
            q[4] += rnd.uniform(-0.2, 0.2)

            self.agent.config = q

        self.agent_controller.stop(self.agent)

        column_names = ["x_target", "y_target", "angle_target", "k1_target", "k2_target", 
                       "x", "y", "angle", "k1", "k2",  
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
        print("Save experiment data")
        df.to_csv('Experiments/Data/reach_target.csv', index=False)



    #////////////////////////////////BUTTONS METHODS//////////////////////////////////////

    def __updatePlots(self) -> None:
        self.gui.clear() # Clear the subplots
        self.gui.plotPaths(self.__paths, self.tracking_area, 'target')
        self.gui.plotMAS()# Plot simplified mas

    def generatePaths(self) -> None:
        self.__paths = {}

        for agent in self.mas.agents:
            path = self.__generatePath(agent)
            self.__paths[agent.id] = path

        self.__updatePlots()
   
    def __generatePath(self, robot: robot2sr.Robot) -> list:
        print('Generate path')

        path = []        
        return path

    def __closeToBorder(self, q):
        result = True

        x_range = self.tracking_area[0]
        y_range = self.tracking_area[1]

        safety_margin = 0.5
        x_safe_range = [x_range[0] + safety_margin, x_range[1] - safety_margin]
        y_safe_range = [y_range[0] + safety_margin, y_range[1] - safety_margin]

        if x_safe_range[0] < q[0] < x_safe_range[1] and y_safe_range[0] < q[1] < y_safe_range[1]:
            result = False

        return result 
    
    def __quadrant(self, q):
        x_range = self.tracking_area[0]
        y_range = self.tracking_area[1]

        x_middle = x_range[0] + (x_range[1] - x_range[0]) / 2
        y_middle = y_range[0] + (y_range[1] - y_range[0]) / 2

        if q[0] > x_middle:
            if q[1] > y_middle:
                return 1
            else:
                return 4
        elif q[1] > y_middle:
            return 2
        else:
            return 3
        
        
    def start(self) -> None:
        pass

    def stop(self) -> None:
        self.mas.stop()

    def quit(self) -> None:
        pass
    
if __name__ == "__main__":
    experiment = Task(Mode.PATH_TRACKING)
    experiment.run()
