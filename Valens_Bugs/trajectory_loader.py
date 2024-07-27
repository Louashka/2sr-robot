import math

import cv2
import cv2.aruco as aruco
import numpy as np
import matplotlib.pyplot as plot

import sys
import nat_net_client as nnc

'''
This is the code for aligning the coordinate loading the trajectory in the camera frame and display it

The algorithm of this code is:
1. Read the OptiTrack markers' positions (unit: meter, coordinate respective to workspace: (-x, z, y))
2. Read the pixel position and the rotation vector of ArUcos
3. Align the center of OptiTrack marker and the ArUco, calculate the positions
    of ArUco in the OptiTrack's coordinate
4. Calculate the distance between two ArUco markers (meter and pixel) to calculate
    the ratio of 'pixel:meter', for converting the trajectory (in meter) to pixel coordinate
5. Read the target trajectory and convert it into pixels
6. Display the trajectory in the camera frames

Relate to the main tasks, multiple related function will be includes, such as undistortion and image cropping
'''
