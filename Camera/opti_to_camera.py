import json
import cv2
import numpy as np
import sys
sys.path.append('D:/Robot 2SR/2sr-swarm-control')
from Motive import nat_net_client as nnc

mocap_data = None
camera_matrix = None
camera_dist = None

json_path = 'Camera/recorded_markers.json'

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

def record() -> None:
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

        if len(labeled_marker_list) != 12:
            continue

        markers = {}
        for marker in labeled_marker_list:
            model_id, marker_id = [int(i) for i in marker.get_id()] 
            marker = {
                'model_id': model_id, 
                'marker_id': marker_id,
                'x': -marker.pos[0], 
                'y': marker.pos[2], 
                'z': marker.pos[1]
            }
            markers[str(model_id) + '.' + str(marker_id)] = marker


        _, frame = cap.read()
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        h, w = frame.shape[:2]
        new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, camera_dist, (w,h), 1, (w,h))
        undistorted_frame = cv2.undistort(frame, camera_matrix, camera_dist, None, new_camera_matrix)

        cv2.imshow("Frame", undistorted_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            filename = 'Camera/Images/snapshot.jpg'
            cv2.imwrite(filename, undistorted_frame)
            print('Writing：', filename)

            with open(json_path, 'w') as f:
                json.dump(markers, f)

            break

    cap.release()
    cv2.destroyAllWindows()


def process():
    markers_data = None
    try:
        with open(json_path, "r") as json_file:
            markers_data = json.load(json_file)
            print('Markers\' data is retrieved.')
    except FileNotFoundError:
        print(f"Error: {json_path} not found.")

    snaphot_path = 'Camera/Images/snapshot_pocessed.jpg'
    snapshot = cv2.imread(snaphot_path)

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(snapshot, cv2.COLOR_BGR2HSV)

    # Define the range of red color in HSV
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    # Create masks for red color
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Minimum area threshold for contours
    min_area = 10
    red_dots = []

    # Draw circles around detected red dots
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = [int(x), int(y)]
            red_dots.append(center)
            radius = int(radius)

    print(f"Detected {len(contours)} red dots")

    # cv2.circle(snapshot, red_dots[10], 2, (0, 255, 0), 2)

    id_order = ['2.2', '0.50001', '2.3', '2.1', '0.50002', '0.50005',
                '0.50006', '0.50007', '0.50008', '1.1', '1.2', '1.3']
    
    object_points = []
    image_points = []
    
    for i in range(len(id_order)):
        marker = markers_data[id_order[i]]
        object_points.extend(np.array([[marker['x'], marker['y'], marker['z']]]))
        image_points.extend(np.array([red_dots[i]]))

    object_points = np.array(object_points, dtype=np.float32)
    image_points = np.array(image_points, dtype=np.float32)

    # Estimate camera pose in global coordinate system
    h, w = snapshot.shape[:2]
    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, camera_dist, (w,h), 1, (w,h))
    R, tvec = estimatePose(object_points, image_points, new_camera_matrix, np.zeros(5))

    for marker in markers_data.values():
        global_point = np.array([marker['x'], marker['y'], marker['z']])  
        camera_point = globalToCamera(global_point, R, tvec)
        image_point = cameraToImage(camera_point, new_camera_matrix)

        # Draw the point on the image
        cv2.circle(snapshot, image_point, 2, (0, 255, 0), -1)

    data_json = {
        "R": R.tolist(),
        "tvec": tvec.flatten().tolist()
    }
    with open('Camera/opti_to_camera_result.json', 'w') as f:
        json.dump(data_json, f)
    
    cv2.imshow("Processed Snapshot", snapshot)
    cv2.waitKey(0)
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

    # print('Start Motive streaming...')
    # startDataListener()

    # print('Start recording...')
    # record()

    print('Process recording...')
    process()