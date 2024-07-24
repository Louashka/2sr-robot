import math

import cv2
import cv2.aruco as aruco
import numpy as np
import matplotlib.pyplot as plot

# Camera
fx = 587.77695912
fy = 587.49352402
cx = 636.67708427
cy = 350.68017254
k1 = 3.58183122e-04
k2 = -2.20971875e-02
p1 = -2.46447770e-05
p2 = 1.46568391e-03
k3 = 6.40455235e-03
cameraMatrix = np.array([[fx, 0, cx],
                         [0, fy, cy],
                         [0, 0, 1]])
dist = np.array([k1, k2, p1, p2, k3])

# aruco markers
markerA = 12
markerB = 10
markerA_position = [0, 0]
markerB_position = [0, 0]
optiTrackA_position = [0, 0]
optiTrackB_position = [0, 0]

marker_length = 0.039   # meters
offset = 0.045    # meters, the offset of the center of aruco to the optitrack marker

ratio = 0.0015587534956059915      # pixel to meter
L = offset / ratio

print(L)

# OptiTrack markers
markerA_ID = 56851
markerA_pos = [0.40, -0.03, 2.17]
markerB_ID = 56845
markerB_pos = [-0.10, 0.04, -1.73]


def undistort(frame):
    h, w = frame.shape[:2]
    mapx, mapy = cv2.initUndistortRectifyMap(cameraMatrix, dist, None, cameraMatrix, (w, h), 5)
    return cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)


def DetectArucoPose(frame):
    global markerA_position, markerB_position, ratio, L, optiTrackA_position, optiTrackB_position

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

    parameters = aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, cameraMatrix, dist)
        print("rvec: ", rvec[0])
        print("tvex: ", tvec[0])
        for i in range(rvec.shape[0]):
            tvecCopy = tvec[i, :, :] + [10., 0, 0]
            # print("tvecCopy", tvecCopy)
            aruco.drawAxis(frame, cameraMatrix, dist, rvec[i, :, :], tvec[i, :, :], 0.03)
            aruco.drawDetectedMarkers(frame, corners, ids)
            # print("rvec[", i, ",: , ：]: ", rvec[i, :, :])

        for i in range(len(ids)):
            if ids[i] == markerA:
                corner_A = corners[i][0]
                center_x = (corner_A[0][0] + corner_A[1][0] + corner_A[2][0] + corner_A[3][0]) / 4
                center_y = (corner_A[0][1] + corner_A[1][1] + corner_A[2][1] + corner_A[3][1]) / 4
                markerA_position = [center_x, center_y]

                # update the parameters if needed
                # distance =math.sqrt((corner_A[0][0] - corner_A[1][0])**2 + (corner_A[0][1] - corner_A[1][1])**2)
                # ratio = marker_length / distance
                # L = offset / ratio

                theta = abs(center_y - corner_A[0][1]) / abs(center_x - corner_A[0][0])
                print(theta)
                x = center_x - L * math.cos(theta)
                y = center_y + L * math.sin(theta)
                optiTrackA_position = [x, y]
                print(optiTrackA_position)
                cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)

            if ids[i] == markerB:
                corner_B = corners[i][0]
                center_x = (corner_B[0][0] + corner_B[1][0] + corner_B[2][0] + corner_B[3][0]) / 4
                center_y = (corner_B[0][1] + corner_B[1][1] + corner_B[2][1] + corner_B[3][1]) / 4
                markerB_position = [center_x, center_y]

                theta = abs(center_y - corner_B[0][1]) / abs(center_x - corner_B[0][0])
                print(theta)
                x = center_x + L * math.cos(theta)
                y = center_y - L * math.sin(theta)
                optiTrackB_position = [x, y]
                print(optiTrackB_position)
                cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)

    cv2.imshow("capture", frame)


cap = cv2.VideoCapture(0)
width = cap.get(3)
height = cap.get(4)
fps = cap.get(5)
print(width, height, fps)  # 640.0 480.0 30.0

# 在这里把摄像头的分辨率修改为和我们标定时使用的一样的分辨率 1280x720
cap.set(3, 1280)
cap.set(4, 720)
width = cap.get(3)
height = cap.get(4)
print(width, height, fps)  # 1280.0 720.0 30.0


while cap.isOpened():
    ret, frame = cap.read()
    # cv2.imshow("original", frame)
    h1, w1 = frame.shape[:2]
    # newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, (h1, w1), 0, (h1, w1))
    # frame = cv2.undistort(frame, cameraMatrix, dist, None, newCameraMatrix)
    undistort_frame = undistort(frame)
    # compare = np.hstack((frame, undistort_frame))
    # cv2.imshow("undistort", undistort_frame)

    DetectArucoPose(undistort_frame)
    print("MarkerA's Position: ", markerA_position)
    print("MarkerB's Position: ", markerB_position)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindwos()