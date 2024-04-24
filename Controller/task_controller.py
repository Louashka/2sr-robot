from enum import Enum
import pandas as pd
import sys
sys.path.append('/Users/lytaura/Documents/PolyU/Research/2SR/Version 1/Multi agent/Control/2sr-swarm-control')
# sys.path.append('D:/Romi-lab/2sr-swarm-control')
from Model import global_var, robot2sr
from View import plotlib
import motive_client, keyboard_controller, grasping_controller, mas_controller
import random as rnd
import numpy as np

# There are 4 task modes
class Mode(Enum):
    MANUAL = 1 # Manual control of a single robot via a keyboard
    GRASP = 2 # Grasp object - approach the target configuration

class Task(keyboard_controller.ActionsHandler):
    def __init__(self, mode: Mode) -> None:
        super().__init__(global_var.OMNI_SPEED, global_var.ROTATION_SPEED, global_var.LU_SPEED)
        self.mode = mode

        self.mocap = motive_client.MocapReader() # Initialise the reader of the tracking data
        self.gui = plotlib.GUI(self) # Initialise GUI

        self.mas = mas_controller.Swarm() # Initialise a multi-agent system (MAS)
        self.manipulandums = None # Initialise a collection of manipulandums

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
        for agent in self.mas.agents:
            self.gui.plotAgent(agent)
        
        # Run the seleceted task mode 
        match self.mode:
            case Mode.MANUAL:
                print('Manual mode')
                self.__manualMode()
            case Mode.GRASP:
                print('Grasp mode')
                self.__graspMode()

        self.gui.window.mainloop() # Start the GUI application

    def __updateConfig(self):
        try:
            # Get the current MAS and manipulandums configuration
            robots, manipulandums = self.mocap.getCurrentConfig()

            for robot in robots:
                if self.mas.getAgentById(robot['id']) is None:
                    new_agent = robot.Robot(robot['id'], robot['x'], robot['y'], robot['theta'], robot['k'])
                    self.mas.agents.append(new_agent)
                else:
                    current_agent = self.mas.getAgentById(robot['id'])
                    current_agent.pose = [robot['x'], robot['y'], robot['theta']]
                    current_agent.k = robot['k']
            
            for manipulandum in manipulandums:
                pass
        except Exception as e:
            print(f"Error occurred: {e}. The robot is stopped!")
            if self.mas is not None:
                self.mas.stop()


    #//////////////////////////////// MANUAL MODE METHODS ////////////////////////////////
    
    def __manualMode(self):
        if len(self.mas.agents) == 0:
            raise Exception('No agents are found!')
        
        if len(self.mas.agents) != 1:
            raise Exception('Wrong number of agents!')
        
        # Handle key events 
        self.gui.window.bind('<KeyPress>', self.__onPress)
        self.gui.window.bind('<KeyRelease>', self.__onRelease)

    # Execute the action according to the keyboard commands
    def __executeAction(self):
        self.__updateConfig()
        self.mas.move(self.v, self.s) # Execute the action by MAS
        # Update the GUI
        for agent in self.mas.agents:
            self.gui.plotAgent(agent)

    def __onPress(self, key) -> None:
        super().onPress(key)
        self.__executeAction()

    def __onRelease(self, key) -> None:
        super().onRelease(key)
        self.__executeAction()

    #///////////////////////////////// GRASP MODE METHODS ////////////////////////////////
    
    def __graspMode(self):
        if len(self.mas.agents) == 0:
            raise Exception('No agents are found!')
        
        if len(self.mas.agents) != 1:
            raise Exception('Wrong number of agents!')


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
