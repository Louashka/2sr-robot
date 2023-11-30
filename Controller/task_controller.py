from enum import Enum
import pandas as pd
import sys
sys.path.append('/Users/lytaura/Documents/PolyU/Research/2SR/Version 1/Multi agent/Control/2sr-swarm-control')
from Model import global_var
from View import plotlib
import motive_client, keyboard_controller, grasping_controller
import random as rnd
import numpy as np

class Mode(Enum):
    MANUAL = 1
    SINGLE = 2
    COLLAB = 3
    COOP = 4

class Task(keyboard_controller.ActionsHandler):
    def __init__(self) -> None:
        super().__init__(global_var.OMNI_SPEED, global_var.ROTATION_SPEED, global_var.LU_SPEED)
        self.__mocap = motive_client.MocapReader() # Initialise the reader of the tracking data
        self.__gui = plotlib.GUI(self) # Initialise the GUI window

        self.__saved_mas = None # Stores the last successfully obtained MAS configuration

    # Execute a task of a given mode
    def run(self, mode) -> None:
        # Check if the task mode is valid
        if not isinstance(mode, Mode):
            raise Exception('Wrong task mode!')

        print('Start Motive streaming')
        self.__mocap.startDataListener() # Start listening data from Motive
        
        print('Start the experiment')
        
        # Run the seleceted task mode 
        match mode:
            case Mode.MANUAL:
                print('Manual mode')
                self.__manualMode()
            case Mode.SINGLE:
                print('Single mode')
                self.__singleMode()
            case Mode.COLLAB:
                print('Collaboration mode')
                self.__collabMode()
            case Mode.COOP:
                print('Cooperation mode')
                self.__coopMode()

        self.__gui.mainloop() # Start the GUI application

    #//////////////////////////////// MANUAL MODE METHODS ////////////////////////////////
    
    def __manualMode(self):
        # Handle key events 
        self.__gui.window.bind('<KeyPress>', self.__onPress)
        self.__gui.window.bind('<KeyRelease>', self.__onRelease )

    # Execute the action according to the keyboard commands
    def __executeAction(self):
        try:
            # Get the current MAS and manipulandums configuration
            self.__mas, self.__manipulandums = self.__mocap.getConfig()

            if self.__mas is not None:
                self.__saved_mas = self.__mas # Update the saved MAS config
                self.__mas.move(self.v, self.s) # Execute the action by MAS
                # Update the GUI
                plotlib.plotMotion(self.__mas, self.__manipulandums)
        except Exception as e:
            print(f"Error occurred: {e}. The robot is stopped!")
            if self.__saved_mas is not None:
                self.__mas.stop()

    def __onPress(self, key) -> None:
        super().onPress(key)
        self.__executeAction()

    def __onRelease(self, key) -> None:
        super().onRelease(key)
        self.__executeAction()

    #//////////////////////////////// SINGLE MODE METHODS ////////////////////////////////
    def __singleMode(self) -> None:
        self.__mas, self.__manipulandums = self.__mocap.getConfig()
        self.__saved_mas = self.__mas

        if len(self.__mas.agents) == 0:
            raise Exception('No agents are found!')
        
        if len(self.__mas.agents) != 1:
            raise Exception('Wrong number of agents!')
        
        if len(self.__manipulandums) > 0:
            raise Warning('Please remove manipulandums!')
        
        self.__gui.window.mainloop()

    def generatePath(self) -> list:
        print('Generate path')

        self.__paths = []
        dt = 0.25

        x_range = [-2, 2]
        y_range = [-2, 2]

        for agent in self.__saved_mas.agents:
            q_current = agent.pose
            path = []

            t = 0
            velocity = [0, rnd.uniform(0.5, 1), rnd.uniform(-1, 1)]

            while t < 6:
                R = np.array([[np.cos(q_current[2]), -np.sin(q_current[2]), 0],
                    [np.sin(q_current[2]), np.cos(q_current[2]), 0], [0, 0, 1]])

                q_dot = R.dot(velocity)
                q_new = q_current + q_dot * dt

                q_current = q_new
                t += dt

                path.append(q_current)

            print(path)
            self.__paths.append(path)

        self.__gui.plotPaths(self.__paths)
 

    #//////////////////////////////// COLLAB MODE METHODS ////////////////////////////////

    def __collabMode(self) -> None:
        # Get the current MAS and manipulandums configuration
        self.__mas, self.__manipulandums = self.__mocap.getConfig()

        if len(self.__manipulandums) != 1:
            raise Exception('Wrong number of manipulandums!')
        
        # Define the cloud shape
        cloud_manipulandum = self.__manipulandums[0]
        cloud_contour_params = self.__extractManipShape(['./Data/manip_contour.csv'])[0]
        cloud_manipulandum.contour_params = cloud_contour_params

        # Define manipulandum's path
        manip_path = self.__generatePath([0, 0, 0])

        # Initialise an optimised grasping model
        grasp = grasping_controller(self.__mas, cloud_manipulandum, manip_path)
        

    def __extractManipShape(self, paths) -> list:
        manip_contour_params = []

        for path in paths:
            contour_df = pd.read_csv(path)
            contour_r = contour_df['distance'].tolist()
            contour_theta = contour_df['phase_angle'].tolist()
            
            manip_contour_params.append([contour_r, contour_theta])

        return manip_contour_params
        
    #//////////////////////////////// COOP MODE METHODS ////////////////////////////////
    
    def __coopMode(self) -> None:
        # Create a 4 manipulandums
        contours = self.__extractManipShape(['', '', '', ''])
    

if __name__ == "__main__":
    experiment = Task()
    experiment.run(Mode.SINGLE)
