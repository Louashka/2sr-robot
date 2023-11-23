import math
import numpy as np
import pandas as pd
import motive_client
import robot_keyboard
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk

OMNI_SPEED = 0.12
ROTATION_SPEED = 1
LU_SPEED = 0.12

# Length and width of the LU  
a = 0.042 
# Distance between LU center and its corner
r = a * math.sqrt(2) / 2 
# Angle between LU orientation and r
alpha = math.radians(-135)

current_pose = 1
pose_markers = None
pose_rigid_bodies = None

root = tk.Tk()
frame = tk.Frame(root)
fig, axs = plt.subplots(nrows=1, ncols=2, figsize = (6, 3))
canvas = FigureCanvasTkAgg(fig, master = frame)

class ManualController(robot_keyboard.ActionsHandler):

    def __init__(self, omni_speed, rotation_speed, lu_speed) -> None:
        super().__init__(omni_speed, rotation_speed, lu_speed)

        self.robot_config = None

    def executeAction(self):
        try:
            self.robot_config = motive_client.getCurrentConfig([pose_markers, pose_rigid_bodies])

            if self.robot_config is not None:
                self.plotMotion()
        except Exception as e:
            print(f"Error occurred: {e}. The robot is stopped!")

    def plotMotion(self):
        # Retrieve motion data 
        markers, all_frames, wheels_global, wheels_bf = self.robot_config 

        # Unpack frames
        LU_head_frame, LU_tail_frame, body_frame, manipulandum_frame = all_frames

        # Update the plot with new motion data 
        axs[0].clear()
        axs[1].clear()

        markers_x = [marker.get('marker_x') for marker in markers.values()]
        markers_y = [marker.get('marker_y') for marker in markers.values()]

        axs[0].scatter(markers_x, markers_y)

        # Plot the block of the head LU
        LU_head_rect = (LU_head_frame[0] + r*np.cos(LU_head_frame[2] + alpha), LU_head_frame[1] + r*np.sin(LU_head_frame[2] + alpha))
        axs[0].add_patch(Rectangle(LU_head_rect, a, a, angle=math.degrees(LU_head_frame[2]), edgecolor='black', facecolor='none'))

        # Plot the block of the tail LU
        LU_tail_centre = (LU_tail_frame[0] + r*np.cos(LU_tail_frame[2] + alpha), LU_tail_frame[1] + r*np.sin(LU_tail_frame[2] + alpha))
        axs[0].add_patch(Rectangle(LU_tail_centre, a, a, angle=math.degrees(LU_tail_frame[2]), edgecolor='black', facecolor='none'))

        # Plot all frames
        for frame in all_frames:
            axs[0].plot(frame[0], frame[1], 'r*')
            axs[0].plot([frame[0], frame[0] + 0.03 * np.cos(frame[2])], [frame[1], frame[1] + 0.03 * np.sin(frame[2])], 'r')

        # Plot wheels
        for wheel in wheels_global:
            axs[0].plot(wheel[0], wheel[1], 'mo', markersize=7)

        # Plot the bridge curve
        vsf_markers = [marker for marker in markers.values() if marker['model_id'] == 0 and marker['rank'] != 6]
        vsf_markers.sort(key=lambda marker: marker['rank'])

        vsf_markers_x = [vsf_marker['marker_x'] for vsf_marker in vsf_markers]
        vsf_markers_y = [vsf_marker['marker_y'] for vsf_marker in vsf_markers]

        axs[0].plot(vsf_markers_x, vsf_markers_y, color='orange')

        # Connect the bridge with the LU's
        vsf_start = (LU_head_frame[0] + r*np.cos(LU_head_frame[2] + alpha - np.pi), LU_head_frame[1] + r*np.sin(LU_head_frame[2] + alpha - np.pi))
        axs[0].plot([vsf_start[0], vsf_markers_x[0]], [vsf_start[1], vsf_markers_y[0]], color='orange')

        vsf_end = (LU_tail_frame[0] + r*np.cos(LU_tail_frame[2] + alpha - np.pi/2), LU_tail_frame[1] + r*np.sin(LU_tail_frame[2] + alpha - np.pi/2))
        axs[0].plot([vsf_end[0], vsf_markers_x[-1]], [vsf_end[1], vsf_markers_y[-1]], color='orange')

        axs[0].axis('equal')

        # PRINT THE ROBOT IN A BODY FRAME

        axs[1].plot(0, 0, 'r*')
        axs[1].plot([0, 0.05 * np.cos(0)], [0, 0.05 * np.sin(0)], 'r')

        for wheel in wheels_bf:
            axs[1].plot(wheel[0], wheel[1], 'mo', markersize=7)
            axs[1].plot([wheel[0], wheel[0] + 0.03 * np.cos(wheel[2])], [wheel[1], wheel[1] + 0.03 * np.sin(wheel[2])], 'r')
        
        axs[1].axis('equal')

        plt.rcParams.update({'font.size': 5})
        plt.tight_layout()

        canvas.draw()

    def onPress(self, key) -> None:
        super().onPress(key)
        self.executeAction()

    def onRelease(self, key) -> None:
        super().onRelease(key)
        print("Key released")

def readData(pose):
    markers_df = pd.read_csv('Data/markers.csv')
    rigid_bodies_df = pd.read_csv('Data/rigid_bodies.csv')

    markers_df_ = markers_df[markers_df["pose"]
                                == pose].drop('pose', axis=1)
    rigid_bodies_df_ = rigid_bodies_df[rigid_bodies_df["pose"] == pose].drop(
        'pose', axis=1)

    markers = {}
    rigid_bodies = {}

    for index, row in markers_df_.iterrows():
        marker = row.to_dict()
        markers[str(marker['model_id']) + '.' + str(marker['marker_id'])] = marker

    for index, row in rigid_bodies_df_.iterrows():
        rigid_body = row.to_dict()
        rigid_bodies[int(rigid_body['id'])] = rigid_body        

    return markers, rigid_bodies

def switchConfig():
    global pose_markers, pose_rigid_bodies, current_pose
    pose_markers, pose_rigid_bodies = readData(current_pose)

    print('Pose: ' + str(current_pose))
    
    if current_pose == 8:
        current_pose = 0
    else:
        current_pose += 1


if __name__ == "__main__":    

    manual_controller = ManualController(OMNI_SPEED, ROTATION_SPEED, LU_SPEED)
    switchConfig()
    
    root.title('Robot motion')
    root.config(background='#fafafa')

    canvas.get_tk_widget().pack()
    frame.pack()

    tk.Button(frame, text='Another config', command = switchConfig).pack(pady=10)

    # toolbar = NavigationToolbar2Tk(canvas, root, pack_toolbar = False)
    # toolbar.update()
    # toolbar.pack(side=tk.BOTTOM, fill=tk.X)

    root.bind('<KeyPress>', manual_controller.onPress)
    root.bind('<KeyRelease>', manual_controller.onRelease )
    root.mainloop()
