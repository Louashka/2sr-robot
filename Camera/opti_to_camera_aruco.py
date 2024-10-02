import math
import json
import cv2
import cv2.aruco as aruco
import numpy as np
import sys
sys.path.append('D:/Robot 2SR/2sr-swarm-control')
from Motive import nat_net_client as nnc

mocap_data = None
camera_matrix = None
camera_dist = None

# for optitrack
def receiveData(value) -> None:
    global mocap_data
    mocap_data = value

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

def parseCameraData(camera_data: dict) -> None:
    global camera_matrix, camera_dist

    fx = camera_data["fx"]
    fy = camera_data["fy"]
    cx = camera_data["cx"]
    cy = camera_data["cy"]
    k1 = camera_data["k1"]
    k2 = camera_data["k2"]
    p1 = camera_data["p1"]
    p2 = camera_data["p2"]
    k3 = camera_data["k3"]

    camera_matrix = np.array([[fx, 0, cx],
                             [0, fy, cy],
                             [0, 0, 1]])
    camera_dist = np.array([k1, k2, p1, p2, k3])

def optiToGlobal(opti_coords: list):
    global_coords = [-opti_coords[0], opti_coords[2], opti_coords[1]]
    
    return global_coords

def shrinkSquare(corners, scale_factor=0.683):
    """Calculate the center of a square given its corner points."""
    center = np.mean(corners, axis=0)
    return center + (corners - center) * scale_factor

def sortCorners(markers):
    center = np.mean(markers, axis=0)
    return sorted(markers, key=lambda p: np.arctan2(p[1] - center[1], p[0] - center[0]))

def sortMarkers(markers_A: list, markers_B: list) -> dict:
    markers_A_corners  = sortCorners(markers_A)
    markers_B_corners  = sortCorners(markers_B)

    # Shrink IR marker squares
    markers_A_shrunk = shrinkSquare(markers_A_corners)
    markers_B_shrunk = shrinkSquare(markers_B_corners)

    sorted_markers_coordinates = {
        10: markers_A_shrunk,
        12: markers_B_shrunk
    }

    return sorted_markers_coordinates

def detectArucoMarkers(frame):
    aruco_dict_type = cv2.aruco.DICT_6X6_250

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco_dict_type)
    parameters = aruco.DetectorParameters_create()
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    return corners, ids

def estimatePose(object_points, image_points, camera_matrix, dist_coeffs):
    _, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec

def globalToCamera(point_global, R, tvec):
    point_global = np.array(point_global).reshape(3, 1)
    point_camera = R @ point_global + tvec
    return point_camera.flatten()

def cameraToImage(point_camera, camera_matrix):
    x, y, z = point_camera
    u = camera_matrix[0, 0] * x / z + camera_matrix[0, 2]
    v = camera_matrix[1, 1] * y / z + camera_matrix[1, 2]
    return (int(u), int(v))


def align() -> None:
    global mocap_data, camera_matrix, camera_dist

    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    while cap.isOpened():

        # Detect IR markers
        if mocap_data is None:
            continue

        labeled_marker_data = mocap_data.labeled_marker_data
        labeled_marker_list = labeled_marker_data.labeled_marker_list

        if len(labeled_marker_list) != 8:
            continue

        markers_A = []
        markers_B = []

        for marker in labeled_marker_list:
            if marker.pos[2] < 0:
                markers_A.append(optiToGlobal(marker.pos))
            else:
                markers_B.append(optiToGlobal(marker.pos))

        sorted_markers_coordinates = sortMarkers(markers_A, markers_B)

        # import matplotlib.pyplot as plt

        # # Plotting the sorted markers
        # plt.figure(figsize=(10, 10))
        # for marker_id, coordinates in sorted_markers_coordinates.items():
        #     x, y, _ = zip(*coordinates)
        #     plt.scatter(x, y, label=f"Marker {marker_id}")
        # plt.axis('equal')  # Ensures the aspect ratio is equal to ensure the plot is not distorted
        # plt.title("Sorted Markers")
        # plt.xlabel("X-axis")
        # plt.ylabel("Y-axis")
        # plt.legend()
        # plt.show()

        # Detect ArUco markers
        _, frame = cap.read()
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        h, w = frame.shape[:2]
        new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, camera_dist, (w,h), 1, (w,h))
        undistorted_frame = cv2.undistort(frame, camera_matrix, camera_dist, None, new_camera_matrix)

        corners, ids = detectArucoMarkers(undistorted_frame)

        # Prepare data for pose estimation
        object_points = []
        image_points = []

        for i in range(len(ids)):
            marker_id = ids[i][0]
            if marker_id in sorted_markers_coordinates:
                object_points.extend(sorted_markers_coordinates[marker_id])
                image_points.extend(corners[i][0])

        object_points = np.array(object_points, dtype=np.float32)
        image_points = np.array(image_points, dtype=np.float32)

        # Estimate camera pose in global coordinate system
        R, tvec = estimatePose(object_points, image_points, new_camera_matrix, np.zeros(5))

        for marker in markers_A + markers_B:
            global_point = np.array(marker)  # Example point in global coordinates
            camera_point = globalToCamera(global_point, R, tvec)
            image_point = cameraToImage(camera_point, new_camera_matrix)

            # Draw the point on the image
            cv2.circle(undistorted_frame, image_point, 5, (0, 255, 0), -1)

        cv2.imshow("Result", undistorted_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            data_json = {
                "R": R.tolist(),
                "tvec": tvec.flatten().tolist()
            }
            with open('Camera/opti_to_camera_result.json', 'w') as f:
                json.dump(data_json, f)

            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    file_path = "Camera/calibration_result.json"
    try:
        with open(file_path, "r") as json_file:
            camera_data = json.load(json_file)
            parseCameraData(camera_data)
            print('Camera data is retrieved.')
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")

    print('Start Motive streaming...')
    startDataListener()
    print('Start aligning the camera and Motive coordinates...')
    align()


    

        