import math

import cv2
import cv2.aruco as aruco
import numpy as np
import matplotlib.pyplot as plot

import sys
import nat_net_client as nnc

data = None

'''
This is the code for loading the trajectory (path of the 2SR agent) in the camera frame
and display it

The algorithm of this code is:
1. Read the OptiTrack markers' positions (unit: meter, coordinate respective to workspace: (-x, z, y))
2. Read the pixel position and the rotation vector of ArUcos
3. Based on the offset of the OptiTrack marker and the ArUco, calculate the positions
    of ArUco in the OptiTrack's coordinate
4. Calculate the distance between two ArUco markers (meter and pixel) to calculate
    the ratio of 'pixel:meter', for converting the trajectory (in meter) to pixel coordinate
5. Read the target trajectory and convert it into pixels
6. Display the trajectory in the camera frames

Relate to the main tasks, multiple related function will be includes, such as undistortion
'''

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

# ArUco Markers
aruco_A_id = 10
aruco_B_id = 12
aruco_A_position = np.array([93, 65])
aruco_B_position = np.array([1151, 619])

marker_length = 0.04    # meters

# OptiTrack Markers
opti_A1_id = 27373
opti_A2_id = 27379
opti_B1_id = 27673
opti_B2_id = 27674
opti_A1_position = np.array([-0.4310126304626465, 0.01978898048400879, 1.2289113998413086])
opti_A2_position = np.array([-0.3793858289718628, 0.018277883529663086, 1.165675163269043])
opti_B1_position = np.array([0.7488584518432617, -0.002796173095703125, -1.2286527156829834])
opti_B2_position = np.array([0.8024942278862, -0.005115032196044922, -1.2898616790771484])

def convert_opti_coordinate_to_workspace(opti_position):    
    '''
    The function to convert the optiTrack coordinate (-x, z, y) to the desired workspace coordinate
    Input: np.array[3]
    Output: the coordinate of the marker in workspace coordinate (x, y, z)
    '''
    workspace_position = np.array([-opti_position[0], opti_position[2], opti_position[1]])
    print("workspace_position: ", workspace_position)
    return workspace_position

opti_A1_position = convert_opti_coordinate_to_workspace(opti_A1_position)
opti_A2_position = convert_opti_coordinate_to_workspace(opti_A2_position)
opti_B1_position = convert_opti_coordinate_to_workspace(opti_B1_position)
opti_B2_position = convert_opti_coordinate_to_workspace(opti_B2_position)

opti_A_position = None      # mid-point of the marker A under opti coordinate
opti_B_position = None      # mid-point of the marker B under opti coordinate

offset = 0.045          # meters, the offset of the center of aruco to the optitrack marker
ratio = 440            # The ratio of meter to pixel, pixel / meter

offset_pixel = offset * ratio      # convert the offset to pixel length
print(offset_pixel)     # Check point

# For undistorting the image
def undistort(frame):
    h, w = frame.shape[:2]
    mapx, mapy = cv2.initUndistortRectifyMap(cameraMatrix, dist, None, cameraMatrix, (w, h), 5)
    return cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

def cal_ratio():
    global ratio

    AB_center_position = cal_distance(marker_A_center, marker_B_center)
    AB_pixel_center_position = cal_distance(aruco_A_position, aruco_B_position)
    ratio = AB_pixel_center_position / AB_center_position

# For ArUco
def DetectArucoPose(frame):
    global aruco_A_position, aruco_B_position, ratio, offset, opti_A_position, opti_B_position

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

    parameters = aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None and len(ids) == 2:
        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, cameraMatrix, dist)
        # print("rvec: ", rvec)
        print("tvex: ", tvec)
        for i in range(rvec.shape[0]):
            tvecCopy = tvec[i, :, :] + [10., 0, 0]
            # print("tvecCopy", tvecCopy)
            aruco.drawAxis(frame, cameraMatrix, dist, rvec[i, :, :], tvec[i, :, :], 0.03)
            aruco.drawDetectedMarkers(frame, corners, ids)
            # print("rvec[", i, ",: , ：]: ", rvec[i, :, :])

        for i in range(len(ids)):
            if ids[i] == aruco_A_id:
                corner_A = corners[i][0]
                # pixel position of marker A
                center_x = (corner_A[0][0] + corner_A[1][0] + corner_A[2][0] + corner_A[3][0]) / 4
                center_y = (corner_A[0][1] + corner_A[1][1] + corner_A[2][1] + corner_A[3][1]) / 4
                markerA_position = [center_x, center_y]
                aruco_A_position = markerA_position
                print("red dot 1 position: ", [int(center_x), int(center_y)])
                cv2.circle(frame, (int(center_x), int(center_y)), 3, (0, 0, 255), -1)

            if ids[i] == aruco_B_id:
                corner_B = corners[i][0]
                center_x = (corner_B[0][0] + corner_B[1][0] + corner_B[2][0] + corner_B[3][0]) / 4
                center_y = (corner_B[0][1] + corner_B[1][1] + corner_B[2][1] + corner_B[3][1]) / 4
                markerB_position = [center_x, center_y]
                aruco_B_position = markerB_position
                print("red dot 1 position: ", [int(center_x), int(center_y)])
                cv2.circle(frame, (int(center_x), int(center_y)), 3, (0, 0, 255), -1)

        # print("Ratio: ", ratio)
        
        edge_01_A = cal_distance(corner_A[0], corner_A[1])
        edge_12_A = cal_distance(corner_A[1], corner_A[2]) 
        edge_23_A = cal_distance(corner_A[2], corner_A[3]) 
        edge_30_A = cal_distance(corner_A[3], corner_A[0]) 
        edge_01_B = cal_distance(corner_B[0], corner_B[1])
        edge_12_B = cal_distance(corner_B[1], corner_B[2]) 
        edge_23_B = cal_distance(corner_B[2], corner_B[3]) 
        edge_30_B = cal_distance(corner_B[3], corner_B[0])
        pixel_length = (edge_01_A+edge_12_A+edge_23_A+edge_30_A+edge_01_B+edge_12_B+edge_23_B+edge_30_B)/8
        ratio = pixel_length / marker_length
        print("ratio: ", ratio) 

    # cv2.imshow("capture", frame)
    return frame

# for optitrack
def receiveData(value) -> None:
    global data
    data = value

def parseArgs(arg_list, args_dict) -> dict:
    arg_list_len = len(arg_list)
    if arg_list_len > 1:
        args_dict["serverAddress"] = arg_list[1]
        if arg_list_len > 2:
            args_dict["clientAddress"] = arg_list[2]
        if arg_list_len > 3:
            if len(arg_list[3]):
                args_dict["use_multicast"] = True
                if arg_list[3][0].upper() == "U":
                    args_dict["use_multicast"] = False
    return args_dict

def startDataListener() -> None:
    options_dict = {}
    options_dict["clientAddress"] = "127.0.0.1"
    options_dict["serverAddress"] = "127.0.0.1"
    options_dict["use_multicast"] = True

    # Start Motive streaning
    options_dict = parseArgs(sys.argv, options_dict)

    streaming_client = nnc.NatNetClient()
    streaming_client.set_client_address(options_dict["clientAddress"])
    streaming_client.set_server_address(options_dict["serverAddress"])
    streaming_client.set_use_multicast(options_dict["use_multicast"])

    streaming_client.mocap_data_listener = receiveData

    streaming_client.run()

def unpackData():
    global opti_A1_position, opti_A2_position, opti_B1_position, opti_B2_position
    if data is not None:

        labeled_marker_data = data.labeled_marker_data

        labeled_marker_list = labeled_marker_data.labeled_marker_list

        for marker in labeled_marker_list:
            model_id, marker_id = [int(i) for i in marker.get_id()]
            marker_info = {'model_id': model_id, 'marker_id': marker_id,
                      'marker_x': marker.pos[0], 'marker_y': marker.pos[1], 'marker_z': marker.pos[2]}
            print(marker_info)

            if marker_id == opti_A1_id:
                opti_position = marker.pos
                opti_A1_position = convert_opti_coordinate_to_workspace(opti_position)
                print("OptiTrack Marker A1 position: ", opti_A1_position)

            if marker_id == opti_A2_id:
                opti_position = marker.pos
                opti_A2_position = convert_opti_coordinate_to_workspace(opti_position)
                print("OptiTrack Marker A2 position: ", opti_A2_position)

            if marker_id == opti_B1_id:
                opti_position = marker.pos
                opti_B1_position = convert_opti_coordinate_to_workspace(opti_position)
                print("OptiTrack Marker B1 position: ", opti_B1_position)

            if marker_id == opti_B2_id:
                opti_position = marker.pos
                opti_B2_position = convert_opti_coordinate_to_workspace(opti_position)
                print("OptiTrack Marker B2 position: ", opti_B2_position)

def cal_distance(A, B):
    distance = np.sqrt((A[0] - B[0])**2 + (A[1] - B[1])**2)
    print("distance: ", distance)
    return distance

def convert_opti_coordinate_to_camera_coordinate(opti_position, aruco_A_position, aruco_B_position, ratio):
    '''
    Input: 
        opti_position: the position of the point under the opti coordinate, and has been convert from the
                        global coordinate to the coordinate whose origin is marker B
        aruco_A_position: the aruco marker A's position, play as the role of referee of the camera frame
        ratio: the ratio of converting meter to pixel, which should be calculated based on the distance
                        (both pixel distance and the meter distance) of two markers
    Output: the camera frame position refer to marker A
    '''
    aruco_A_to_B_x = aruco_B_position[0] - aruco_A_position[0]
    aruco_A_to_B_y = aruco_B_position[1] - aruco_A_position[1]
    pixel_position = opti_position * ratio   # directly convert the meter length to pixel length
    print("pixel_position: ", pixel_position)
    marker_pixel_postion_x = aruco_A_position[0] + aruco_A_to_B_x - pixel_position[1]
    marker_pixel_postion_y = aruco_A_position[1] + aruco_A_to_B_y - pixel_position[0]
    marker_pixel_postion = np.array([marker_pixel_postion_x, marker_pixel_postion_y])

    return marker_pixel_postion

if __name__ == "__main__":
    count = 0
    arucoA_positions = np.zeros((2, 10))
    arucoB_positions = np.zeros((2, 10))
    optiA_positions = np.zeros((2, 10))
    optiB_positions = np.zeros((2, 10))

    cap = cv2.VideoCapture(0)
    width = cap.get(3)
    height = cap.get(4)
    fps = cap.get(5)
    # print(width, height, fps)  # 640.0 480.0 30.0

    # 在这里把摄像头的分辨率修改为和我们标定时使用的一样的分辨率 1280x720
    cap.set(3, 1280)
    cap.set(4, 720)
    width = cap.get(3)
    height = cap.get(4)
    print(width, height, fps)  # 1280.0 720.0 30.0

    startDataListener()

    while cap.isOpened():
        ret, frame = cap.read()
        # cv2.imshow("original", frame)
        h1, w1 = frame.shape[:2]
        # newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, (h1, w1), 0, (h1, w1))
        # frame = cv2.undistort(frame, cameraMatrix, dist, None, newCameraMatrix)
        undistort_frame = undistort(frame)
        # undistort_frame = frame
        # compare = np.hstack((frame, undistort_frame))
        # cv2.imshow("undistort", undistort_frame)

        if count < 10:
            DetectArucoPose(undistort_frame)
            print("ArUco Marker A's Position: ", aruco_A_position)
            print("ArUco Marker B's Position: ", aruco_B_position)

            unpackData()

            arucoA_positions[0][count] = aruco_A_position[0]
            arucoA_positions[1][count] = aruco_A_position[1]

            arucoB_positions[0][count] = aruco_B_position[0]
            arucoB_positions[1][count] = aruco_B_position[1]

            optiA_positions[0][count] = (opti_A1_position[0] + opti_A2_position[0]) / 2
            optiA_positions[1][count] = (opti_A1_position[1] + opti_A2_position[1]) / 2

            optiB_positions[0][count] = (opti_B1_position[0] + opti_B2_position[0]) / 2
            optiB_positions[1][count] = (opti_B1_position[1] + opti_B2_position[1]) / 2

            count += 1
            print(count)

        else:
            # print(arucoA_positions)
            # print(optiA_positions)
            aruco_A_position = np.array([np.average(arucoA_positions[0]), np.average(arucoA_positions[1])])
            aruco_B_position = np.array([np.average(arucoB_positions[0]), np.average(arucoB_positions[1])])

            opti_A_position = np.array([np.average(optiA_positions[0]), np.average(optiA_positions[1])])
            opti_B_position = np.array([np.average(optiB_positions[0]), np.average(optiB_positions[1])])
            print("optiTrack marker set A: ", opti_A_position)
            print("optiTrack marker set B: ", opti_B_position)

            undistort_frame = DetectArucoPose(undistort_frame)
            
            '''In case of being confused, here is a graph of how to convert the coordinate
            . ------> x (camera)
            |       -----------------------------------------
            |       | A                                     |
            v       |                         x (workspace)^|
    y (camera)      |                                      ||
                    |                    y(workspace)<----.B|
                    -----------------------------------------

            workspace origin (in meter) = x_position of marker B, y_position of marker B
            use the function '' to convert
            '''

            coordinate_origin = opti_B_position
            print("coordinate_origin: ", coordinate_origin)
            # print("markerA_pos - camera_zero: ", markerA_pos - camera_zero)
            markerA_pos = np.array([opti_A_position[0] - coordinate_origin[0], opti_A_position[1] - coordinate_origin[1]])
            markerB_pos = np.array([opti_B_position[0] - coordinate_origin[0], opti_B_position[1] - coordinate_origin[1]])  # (0, 0)
            print("markerA_pos: ", markerA_pos)
            print("markerB_pos: ", markerB_pos)
            print("aruco marker A position: ", aruco_A_position)
            print("aruco marker B position: ", aruco_B_position)

            # calculate the distance in meters and the pixel distance of two coordinates, and calculate the ratio
            aruco_distance = cal_distance(aruco_A_position, aruco_B_position)
            opti_distance = cal_distance(markerA_pos, markerB_pos)

            # ratio = aruco_distance / opti_distance      # pixel to meter
            print("Ratio： ", ratio)
            print("aruco_x_distance: ", aruco_B_position[0] - aruco_A_position[0])
            print("aruco_y_distance: ", aruco_B_position[1] - aruco_A_position[1])
            # convert the opti coordinate back to camera coordinate
            optiA_cam_pos = convert_opti_coordinate_to_camera_coordinate(markerA_pos, aruco_A_position, aruco_B_position, ratio)
            optiB_cam_pos = convert_opti_coordinate_to_camera_coordinate(markerB_pos, aruco_A_position, aruco_B_position, ratio)
            print("optiA_cam_pos: ", optiA_cam_pos)
            print("optiB_cam_pos: ", optiB_cam_pos)
            cv2.circle(undistort_frame, (int(optiA_cam_pos[0]), int(optiA_cam_pos[1])), 3, (0, 255, 0), -1)
            cv2.circle(undistort_frame, (int(optiB_cam_pos[0]), int(optiB_cam_pos[1])), 3, (0, 255, 0), -1)
            cv2.imshow("result", undistort_frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindwos()