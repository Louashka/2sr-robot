from enum import Enum
import pandas as pd
import sys
# sys.path.append('/Users/lytaura/Documents/PolyU/Research/2SR/Version 1/Multi agent/Control/2sr-swarm-control')
sys.path.append('D:/Romi-lab/2sr-swarm-control')
from Model import global_var
from View import plotlib
import motive_client, keyboard_controller, grasping_controller, mas_controller
import random as rnd
import numpy as np

class Mode(Enum):
    MANUAL = 1
    SINGLE = 2
    COLLAB = 3
    COOP = 4

class Task(keyboard_controller.ActionsHandler):
    def __init__(self, mode) -> None:
        super().__init__(global_var.OMNI_SPEED, global_var.ROTATION_SPEED, global_var.LU_SPEED)
        self.mode = mode

        self.__mocap = motive_client.MocapReader() # Initialise the reader of the tracking data
        self.__gui = plotlib.GUI(self) # Initialise GUI

        self.__mas = None
        self.__manipulandums = None

        self.__tracking_area = [[-1, 3], [-1, 3]]

    @property
    def mode(self) -> Mode:
        return self.__mode
    
    @mode.setter
    def mode(self, value) -> None:
        if not isinstance(value, Mode):
            raise Exception('Wrong task mode!')
        self.__mode = value

    # Execute a task of a given mode
    def run(self) -> None:
        # Check if the task mode is valid

        print('Start Motive streaming')
        self.__mocap.startDataListener() # Start listening data from Motive
        
        print('Start the experiment')
        
        # Run the seleceted task mode 
        match self.mode:
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

        self.__gui.window.mainloop() # Start the GUI application

    #//////////////////////////////// MANUAL MODE METHODS ////////////////////////////////
    
    def __manualMode(self):
        # Handle key events 
        self.__gui.window.bind('<KeyPress>', self.__onPress)
        self.__gui.window.bind('<KeyRelease>', self.__onRelease)

    # Execute the action according to the keyboard commands
    def __executeAction(self):
        try:
            # Get the current MAS and manipulandums configuration
            markers, self.__mas, self.__manipulandums = self.__mocap.getCurrentConfig()

            if self.__mas is not None:
                self.__mas.move(self.v, self.s) # Execute the action by MAS
                # Update the GUI
                self.__gui.plotMarkers(markers)
                for agent in self.__mas.agents:
                    self.__gui.plotAgent(agent)
        except Exception as e:
            print(f"Error occurred: {e}. The robot is stopped!")
            if self.__mas is not None:
                self.__mas.stop()

    def __onPress(self, key) -> None:
        super().onPress(key)
        self.__executeAction()

    def __onRelease(self, key) -> None:
        super().onRelease(key)
        self.__executeAction()

    #//////////////////////////////// SINGLE MODE METHODS ////////////////////////////////
    def __singleMode(self) -> None:
        markers, self.__mas, self.__manipulandums = self.__mocap.getCurrentConfig()

        if len(self.__mas.agents) == 0:
            raise Exception('No agents are found!')
        
        if len(self.__mas.agents) != 1:
            raise Exception('Wrong number of agents!')
        
        if len(self.__manipulandums) > 0:
            raise Warning('Please remove manipulandums!')
        
        self.__gui.plotMarkers(markers)
        
        for agent in self.__mas.agents:
            self.__gui.plotAgent(agent)
        
        self.__gui.window.mainloop()

    #//////////////////////////////// COLLAB MODE METHODS ////////////////////////////////

    def __collabMode(self) -> None:
        # Get the current MAS and manipulandums configuration
        self.__mas, self.__manipulandums = self.__mocap.getCurrentConfig()

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

    #////////////////////////////////BUTTONS METHODS//////////////////////////////////////

    def generatePaths(self) -> None:
        self.__paths = []

        if self.__mas is not None:
            for agent in self.__mas.agents:
                # self.__paths.append(self.__generatePath(agent))

                self.__gui.plotAgent(agent)

   
    def __generatePath(self, agent) -> list:
        print('Generate path')

        path = []
        q_current = agent.config

        t = 0 
        t_count = 0
        dt = 0.25
        velocity = [0, rnd.uniform(0.5, 1), rnd.uniform(-1, 0), 0, 0]
        avoid_border = False

        while t < 6:
            if t_count >= 2:
                velocity = [0, rnd.uniform(0.5, 1), rnd.uniform(-1, 1), 0, 0]
                t_count = 0

            R = np.array([[np.cos(q_current[2]), -np.sin(q_current[2]), 0, 0, 0],
                [np.sin(q_current[2]), np.cos(q_current[2]), 0, 0, 0], [0, 0, 1, 0, 0], [0]*5, [0]*5])

            q_dot = R.dot(velocity)
            q_current += q_dot * dt

            if self.__closeToBorder(q_current):
                quadrant = self.__quadrant(q_current)
                velocity[1] = 0.2

                if not avoid_border:
                    if q_dot[0] > 0 and  q_dot[1] > 0 or q_dot[0] < 0 and  q_dot[1] < 0:
                        velocity[2] = -1
                    elif q_dot[0] < 0 and  q_dot[1] > 0 or q_dot[0] > 0 and  q_dot[1] < 0:
                        velocity[2] = 1
                    elif q_dot[0] == 0:
                        if quadrant == 1 or quadrant == 3:
                            velocity[2] = 1
                        else:
                            velocity[2] = -1
                    elif q_dot[1] == 0:
                        if quadrant == 1 or quadrant == 3:
                            velocity[2] = -1
                        else:
                            velocity[2] = 1
                avoid_border = True
                t_count = 0
            elif avoid_border:
                velocity[1] = rnd.uniform(0.5, 1)
                velocity[2] = rnd.uniform(-1, 1)
                avoid_border = False

            path.append(q_current.tolist())
            self.__gui.plotPath(path, self.__tracking_area, 'target')
            
            t += dt
            t_count += dt
        
        return path

    def __closeToBorder(self, q):
        result = True

        x_range = self.__tracking_area[0]
        y_range = self.__tracking_area[1]

        safety_margin = 0.5
        x_safe_range = [x_range[0] + safety_margin, x_range[1] - safety_margin]
        y_safe_range = [y_range[0] + safety_margin, y_range[1] - safety_margin]

        if x_safe_range[0] < q[0] < x_safe_range[1] and y_safe_range[0] < q[1] < y_safe_range[1]:
            result = False

        return result 
    
    def __quadrant(self, q):
        x_range = self.__tracking_area[0]
        y_range = self.__tracking_area[1]

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
        if self.mode == Mode.SINGLE:
            pass

    def stop(self) -> None:
        pass

    def quit(self) -> None:
        pass
    

if __name__ == "__main__":
    experiment = Task(Mode.MANUAL)
    experiment.run()
