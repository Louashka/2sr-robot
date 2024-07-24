import math

import cv2
import cv2.aruco as aruco
import numpy as np
import matplotlib.pyplot as plot

import sys
import nat_net_client as nnc

data = None

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
markerA_position = np.array([0, 0])
markerB_position = np.array([0, 0])
# optiTrackA_position = np.array([0, 0])
# optiTrackB_position = np.array([0, 0])

marker_length = 0.039   # meters
offset = 0.045    # meters, the offset of the center of aruco to the optitrack marker

ratio = 0.0022      # pixel to meter
L = offset / ratio

print(L)

# OptiTrack markers
markerA_ID = 50003
markerA_pos = np.array([1.543, -0.029, -0.497])
markerB_ID = 50005
markerB_pos = np.array([-1.823, 0.090, 0.803])


# For aruco
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
        # print("rvec: ", rvec)
        # print("tvex: ", tvec)
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
                # cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)

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
                # cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)

        print("Ratio: ", ratio)

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
    global markerA_pos, markerB_pos
    if data is not  None:

        labeled_marker_data = data.labeled_marker_data

        labeled_marker_list = labeled_marker_data.labeled_marker_list

        for marker in labeled_marker_list:
            model_id, marker_id = [int(i) for i in marker.get_id()]
            marker_info = {'model_id': model_id, 'marker_id': marker_id,
                      'marker_x': marker.pos[0], 'marker_y': marker.pos[1], 'marker_z': marker.pos[2]}
            print(marker_info)

            if marker_id == markerA_ID:
                markerA_pos = [marker.pos[0], marker.pos[1], marker.pos[2]]
                print("markerA_pos: ", markerA_pos)

            if marker_id == markerB_ID:
                markerB_pos = [marker.pos[0], marker.pos[1], marker.pos[2]]
                print("markerB_pos: ", markerB_pos)


def cal_distance(A, B):
    distance = np.sqrt((A[0] - B[0])**2 + (A[1] - B[1])**2)
    return distance


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
        # compare = np.hstack((frame, undistort_frame))
        # cv2.imshow("undistort", undistort_frame)

        if count < 10:
            DetectArucoPose(undistort_frame)
            print("MarkerA's Position: ", markerA_position)
            print("MarkerB's Position: ", markerB_position)

            unpackData()

            arucoA_positions[0][count] = markerA_position[0]
            arucoA_positions[1][count] = markerA_position[1]

            arucoB_positions[0][count] = markerB_position[0]
            arucoB_positions[1][count] = markerB_position[1]

            optiA_positions[0][count] = markerA_pos[0]
            optiA_positions[1][count] = markerA_pos[2]

            optiB_positions[0][count] = markerB_pos[0]
            optiB_positions[1][count] = markerB_pos[2]

            count += 1
            print(count)

        # print(arucoA_positions)
        # print(optiA_positions)
        markerA_position = np.array([np.average(arucoA_positions[0]), np.average(arucoA_positions[1])])
        markerB_position = np.array([np.average(arucoB_positions[0]), np.average(arucoB_positions[1])])

        markerA_pos = np.array([np.average(optiA_positions[0]), np.average(optiA_positions[1])])
        markerB_pos = np.array([np.average(optiB_positions[0]), np.average(optiB_positions[1])])
        print("markerA_pos: ", markerA_pos)
        print("markerB_pos: ", markerB_pos)

        undistort_frame = DetectArucoPose(undistort_frame)

        camera_zero = np.array([markerA_pos[0], markerB_pos[1]])
        print("camera_zero: ", camera_zero)
        # print("markerA_pos - camera_zero: ", markerA_pos - camera_zero)
        markerA_pos = np.array([camera_zero[0] - markerA_pos[0], camera_zero[1] - markerA_pos[1]])
        markerB_pos = np.array([camera_zero[0] - markerB_pos[0], camera_zero[1] - markerB_pos[1]])

        aruco_distance = cal_distance(markerA_position, markerB_position)
        opti_distance = cal_distance(markerA_pos, markerB_pos)

        ratio = aruco_distance / opti_distance

        print("Ratio： ", ratio)

        optiA_cam_pos = markerA_pos * ratio + 50
        optiB_cam_pos = markerB_pos * ratio + 50
        print("optiA_cam_pos: ", optiA_cam_pos)
        print("optiB_cam_pos: ", optiB_cam_pos)
        cv2.circle(undistort_frame, (int(optiA_cam_pos[0]), int(optiA_cam_pos[1])), 3, (0, 0, 255), -1)
        cv2.circle(undistort_frame, (int(optiB_cam_pos[0]), int(optiB_cam_pos[1])), 3, (0, 0, 255), -1)
        cv2.imshow("result", undistort_frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindwos()