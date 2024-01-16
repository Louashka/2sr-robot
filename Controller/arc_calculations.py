import sys
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import math
from Model import global_var

# Step 1: get the position of each points
# in this step, tempery use the matplotlib method to plot the screenshot of pose6.png
# the manually read the position of six markers, with result:
# Segment 1 (three markers close to the locomotion unit):
#       [107, 151], [75, 149], [50, 130]
# Segment 2 (three markers distance from the locomotion unity):
#       [42, 100], [46, 84], [80, 60]
# where
# -------> x
# |
# |
# v
# y

# # Code:
# img = cv2.imread("../Data/Screenshots/pose6.png")
#
# plt.figure(figsize=(12, 7))
# plt.subplot(111), plt.title("img"), plt.imshow(img)
# plt.show()


# x1 = [107, 75, 50]
# y1 = [151, 149, 130]
# x2 = [42, 46, 80]
# y2 = [100, 84, 60]
#
# # Create the plot
# fig, ax = plt.subplots()
# ax.invert_yaxis()  # Invert the y-axis to match the camera frame
# ax.plot(x1, y1, 'ro-', label='Segment 1')  # Plot segment 1
# ax.plot(x2, y2, 'bo-', label='Segment 2')  # Plot segment 2
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_title('Camera Frame Coordinates')
# ax.legend()
#
# # Display the plot
# plt.show()


def calculate_curvature(segment_points, segment_length):
    # for the segment_points, the input should be in the following format:
    # [end1, center, end2]
    # point A, point B, point C

    # segment_length is in meter, and the value is 0.077m

    # Calculate the distances between the points
    c = math.sqrt(
        (segment_points[0][0] - segment_points[1][0]) ** 2 + (segment_points[0][1] - segment_points[1][1]) ** 2)
    a = math.sqrt(
        (segment_points[2][0] - segment_points[1][0]) ** 2 + (segment_points[2][1] - segment_points[1][1]) ** 2)
    b = math.sqrt(
        (segment_points[0][0] - segment_points[2][0]) ** 2 + (segment_points[0][1] - segment_points[2][1]) ** 2)

    # Calculate the radius
    alpha = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))
    # print("alpha", alpha)
    r = a / (2 * math.sin(alpha))
    # print("radius", r)

    # Calculate the central angle
    theta = segment_length / r
    # print("central angle", theta)

    # Calculate the curvature, should be positive value all time (need to be comfirn)
    curvature = theta / segment_length

    # Determine the sign by slope
    # Normalize the segment and compute the reletive angle to the end1
    theta1 = math.atan2(segment_points[1][0] - segment_points[0][0], segment_points[1][1] - segment_points[0][1])
    theta2 = math.atan2(segment_points[2][0] - segment_points[0][0], segment_points[2][1] - segment_points[0][1])

    # print("theta1: ", theta1)
    # print("theta2: ", theta2)

    if theta1 > theta2:
        curvature = - curvature

    return curvature

# Define the key points for each segment
# pose 6
segment_1 = [[-0.254846364, -0.185427323], [-0.301700532, -0.218516394], [-0.305579782, -0.239921033]]  # end1, center, end2
segment_2 = [[-0.293332338, -0.279824108], [-0.260157853, -0.305335701], [-0.218587384, -0.308181256]]    # end1, center, end2



# Define the segment length in meters
segment_length = global_var.L_VSS

# Calculate the curvature for each segment
curvature_1 = calculate_curvature(segment_1, segment_length)
curvature_2 = calculate_curvature(segment_2, segment_length)

print("Curvature for segment 1:", curvature_1)
print("Curvature for segment 2:", curvature_2)


# def calculate_arc_center_radius(end1, middle, end2):
#     # Step 1: Calculate the midpoint
#     midpoint_x = (end1[0] + end2[0]) / 2
#     midpoint_y = (end1[1] + end2[1]) / 2
#
#     # Step 2: Calculate the slope
#     slope = (end2[1] - end1[1]) / (end2[0] - end1[0])
#
#     # Step 3: Calculate the perpendicular slope
#     perpendicular_slope = -1 / slope
#
#     # Step 4: Calculate the second midpoint
#     midpoint2_x = (midpoint_x + middle[0]) / 2
#     midpoint2_y = (midpoint_y + middle[1]) / 2
#
#     # Step 5: Calculate the x-coordinate of the center
#     center_x = (midpoint2_y - midpoint_y + perpendicular_slope * midpoint2_x - perpendicular_slope * midpoint_x) / (
#                 2 * perpendicular_slope)
#
#     # Step 6: Calculate the y-coordinate of the center
#     center_y = (-1 / perpendicular_slope) * (center_x - midpoint_x) + midpoint_y
#
#     # Step 7: Calculate the radius
#     radius = math.sqrt((center_x - end1[0]) ** 2 + (center_y - end1[1]) ** 2)
#
#     return (center_x, center_y, radius)
#
#
# # Example usage
# end1 = [0, 0]
# middle = [2, 4]
# end2 = [4, 1]
#
# center, radius = calculate_arc_center_radius(end1, middle, end2)
# print(f"Center: {center}, Radius: {radius}")
