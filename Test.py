import pandas as pd
import motive_client
import robot_keyboard
from PyQt5 import QtCore, QtWidgets
from tkinter import *
from pynput import keyboard

OMNI_SPEED = 0.12
ROTATION_SPEED = 1
LU_SPEED = 0.12

class ManualController(robot_keyboard.ActionsHandler):

    def __init__(self, omni_speed, rotation_speed, lu_speed) -> None:
        super().__init__(omni_speed, rotation_speed, lu_speed)

        self.mocap_data = None
        self.robot_config = None

    def executeAction(self):
        try:
            self.robot_config = motive_client.getRobotConfig([self.mocap_data])

            if self.robot_config is not None:
                print("Move robot")
        except Exception as e:
            print(f"Error occurred: {e}. The robot is stopped!")

    def onPress(self, key) -> None:
        super().onPress(key)
        # self.executeAction()

    def onRelease(self, key) -> None:
        super().onRelease(key)
        print("Key released")

if __name__ == "__main__":
    markers_df = pd.read_csv('Data/markers.csv')
    rigid_bodies_df = pd.read_csv('Data/rigid_bodies.csv')

    for pose in range(1, 2):
        markers_df_ = markers_df[markers_df["pose"]
                                 == pose].drop('pose', axis=1)
        rigid_bodies_df_ = rigid_bodies_df[rigid_bodies_df["pose"] == pose].drop(
            'pose', axis=1)

        markers = {}
        rigid_bodies = {}

        for index, row in markers_df_.iterrows():
            marker = row.to_dict()
            markers[marker['marker_id']] = marker

        for index, row in rigid_bodies_df_.iterrows():
            rigid_body = row.to_dict()
            rigid_bodies[rigid_body['id']] = rigid_body

        motive_client.getRobotConfig([markers, rigid_bodies])

    manual_controller = ManualController(OMNI_SPEED, ROTATION_SPEED, LU_SPEED)

    # Create an instance of tkinter frame or window
    win= Tk()

    # Set the size of the window
    win.geometry("700x350")

    # Create a label widget to add some text
    label= Label(win, text= "", font= ('Helvetica 17 bold'))
    label.pack(pady= 50)

    print("Start!")
    
    win.bind('<KeyPress>', manual_controller.onPress)
    win.bind('<KeyRelease>', manual_controller.onRelease )
    win.mainloop()
    # with keyboard.Listener(on_press=manual_controller.onPress, on_release=manual_controller.onRelease, suppress=True) as listener:
    #     listener.join()
