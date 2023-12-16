from Controller import path
from Model import global_var as gv
import numpy as np
import tkinter as tk
import tkinter.font as font
import matplotlib.pyplot as plt

class Task:
    def __init__(self) -> None:
        self.path = None        

    def generatePath():
        pass

    def start():
        pass

    def stop():
        pass

    def quit():
        pass


class GUI:
    def __init__(self, task: Task) -> None:
        self.__task = task
        self.__window = tk.Tk()
        self.__window.title('Path Tracking')
        self.__window.config(background='#fafafa')

        self.__createSideBar()

    def show(self):
        self.__window.mainloop()

    def __createSideBar(self) -> None:
        self.__side_bar_frame = tk.Frame(self.__window)
        self.__side_bar_frame.pack(fill=tk.Y, side=tk.RIGHT)

        self.__buttons_frame = tk.Frame(self.__side_bar_frame)
        self.__buttons_frame.pack(fill=tk.NONE, expand=True)

        self.__buttons = {}
        
        self.__buttons['gen_path'] = tk.Button(self.__buttons_frame, text='Generate paths', command=self.__task.generatePath)
        self.__buttons['start'] = tk.Button(self.__buttons_frame, text='Start', command=self.__task.start)
        self.__buttons['stop'] = tk.Button(self.__buttons_frame, text='Stop', command=self.__task.stop)
        self.__buttons['exit'] = tk.Button(self.__buttons_frame, text='Exit', command=self.__task.quit)
        
        for button in self.__buttons.values():
            button['font'] = font.Font(size=26)
            button.pack(fill=tk.X, padx=20, pady=8)




if __name__ == "__main__":
    # mocap = motive_client.MocapReader()
    # mocap.startDataListener()
    # task = Task()
    # gui = GUI(task)
    # gui.show()

    path_curve, cp = path.generateCurve()
    
    robot_initial_location = path_curve[0,:]
    robot_goal = path_curve[-1,:]

    robot_current_pose = [0, 0, 0]

    desired_linear_velocity = gv.OMNI_SPEED
    max_angular_velocity = gv.ROTATION_SPEED

    lool_ahead_distance = 0.3

    goal_radius = 0.1
    distance_to_goal = np.linalg.norm(robot_goal - robot_initial_location)

    while distance_to_goal > goal_radius:
        pass



    plt.plot(path_curve[:, 0], path_curve[:, 1])
    plt.plot(cp[:, 0], cp[:, 1], 'ro:')
    plt.axis('equal')
    plt.show()