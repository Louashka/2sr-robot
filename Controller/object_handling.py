# Make grasping first:
# 1. Select the closest point on the object's contour
# 2. Approach the point (misth some pre margin) in a rigid mode
# 3. Grasp the object in a soft mode SM3

import sys
sys.path.append('D:/Robot 2SR/2sr-swarm-control')
from Model import global_var as gv, robot2sr as rsr, manipulandum
import motive_client, robot2sr_controller as rsr_ctrl, camera_optitrack_synchronizer as cos
import pandas as pd
import threading
from datetime import datetime
import numpy as np
import pandas as pd

mocap = motive_client.MocapReader()
rgb_camera = cos.Aligner()

markers = None

agent: rsr.Robot = None
agent_controller = rsr_ctrl.Controller()

cheescake_contour = None
manip: manipulandum.Shape = None
manip_target: manipulandum.Shape = None

target_manip_pos = [0.25, 1.04]
closest_s_index = None

def extractManipShape(path) -> list:
    contour_df = pd.read_csv(path)
    contour_r = contour_df['radius'].tolist()
    contour_theta = contour_df['phase_angle'].tolist()

    manip_contour_params = [contour_r, contour_theta]

    return manip_contour_params


def updateConfig():
    global agent, manip, markers
    # Get the current MAS and manipulandums configuration
    agent_config, manip_config, markers, msg = mocap.getAgentConfig(manip_exp_n=1)

    if agent_config and manip_config:
        if agent:
            agent.pose = [agent_config['x'], agent_config['y'], agent_config['theta']]
            agent.k1 = agent_config['k1']
            agent.k2= agent_config['k2']
        else:
            agent = rsr.Robot(agent_config['id'], agent_config['x'], agent_config['y'], agent_config['theta'], agent_config['k1'], agent_config['k2'])

        agent.head.pose = agent_config['head']
        agent.tail.pose = agent_config['tail']

        # print(f'Agent\'s pose: {agent.pose}')

        manip_pose = [manip_config['x'], manip_config['y'], manip_config['theta']]
        if manip:
            manip.pose = manip_pose
        else:
            manip = manipulandum.Shape(manip_config['id'], manip_pose, cheescake_contour)
        # print(f'Manip\'s pose: {manip.pose}')
    else:
        print('Agent and/or manipulandum is not detected! ' + msg)

def updateConfigLoop():
    global rgb_camera, agent, markers
    while True:
        updateConfig()
        if rgb_camera is not None and agent is not None and manip_target is not None:
            rgb_camera.current_config = agent.config
            rgb_camera.markers = markers

            nearest_point = getNearestPoint()
            rgb_camera.contact_point = nearest_point[:2]
            rgb_camera.manip_target_pos = manip_target.position
            rgb_camera.target_contact_point = manip_target.getPoint(closest_s_index)

def getNearestPoint():
    global closest_s_index
    # Find the closest point on manip.contour to agent.position
    agent_pos = np.array(agent.position)  # Only consider x and y coordinates
    # manip_contour = manip.parametric_contour.T  # Transpose to get a list of [x, y] points
    s_array, manip_contour = manip.parametric_contour
    manip_contour = manip_contour.T
    
    # Calculate distances from agent to all points on the contour
    distances = np.linalg.norm(manip_contour - agent_pos, axis=1)
    
    # Find the index of the minimum distance
    closest_point_index = np.argmin(distances)
    closest_s_index = s_array[closest_point_index]
    
    # Get the closest point
    closest_point = manip_contour[closest_point_index]
    orientation = manip.getTangent(s_array[closest_point_index])
    target_pose = [closest_point[0], closest_point[1], orientation]

    return target_pose + [14, 14]
    
    
def updateContour():
    global rgb_camera, manip
    # Convert cheescake_contour to phase angles and radiuses with respect to manip.pose
    if rgb_camera.cheescake_contour and manip:
        manip_x, manip_y, manip_theta = manip.pose
        phase_angles = []
        radiuses = []
        
        for point in rgb_camera.cheescake_contour:
            # Translate the point relative to manip's position
            dx = point[0] - manip_x
            dy = point[1] - manip_y
            
            # Rotate the point to align with manip's orientation
            rotated_x = dx * np.cos(-manip_theta) - dy * np.sin(-manip_theta)
            rotated_y = dx * np.sin(-manip_theta) + dy * np.cos(-manip_theta)
            
            # Calculate phase angle and radius
            phase_angle = np.arctan2(rotated_y, rotated_x)
            radius = np.sqrt(rotated_x**2 + rotated_y**2)
            
            phase_angles.append(phase_angle)
            radiuses.append(radius)

        # Save phase angles and radiuses to CSV file
        csv_file_path = 'Experiments/Data/Contours/cheescake_contour.csv'
        # Create a DataFrame from the phase angles and radiuses
        df = pd.DataFrame({'phase_angle': phase_angles, 'radius': radiuses})

        # Save the DataFrame to a CSV file
        df.to_csv(csv_file_path, index=False)
        print(f"Cheesecake contour data saved to {csv_file_path}")

def closeToGoal(current, target):
    status = True

    # Calculate Euclidean distance between current and target (x, y)
    distance = np.linalg.norm(np.array(current[:2]) - np.array(target[:2]))
    
    # Calculate absolute difference between current and target theta
    theta_difference = abs(current[2] - target[2])
    
    # Define thresholds for position and orientation
    distance_threshold = 0.02  # 5 cm
    theta_threshold = 0.1  # about 5.7 degrees
    
    # Check if both position and orientation are within thresholds
    if distance > distance_threshold or theta_difference > theta_threshold:
        status = False
    
    print(f"Distance to goal: {distance:.3f} m")
    print(f"Orientation difference: {theta_difference:.3f} rad")
    print()

    return status

def closeToShape(current_k, target_k):
    status = True

    k1_diff = abs(current_k[0] - target_k[0])
    k2_diff = abs(current_k[1] - target_k[1])

    if k1_diff > 5 or k2_diff > 5:
        status = False

    return status

def goToPoint(point, v_prev):
    s = [0, 0]
    finish = False

    if closeToGoal(agent.pose, point) or rgb_camera.finish:
        v_rigid = [0.0] * 3
        finish = True
    else:
        v_rigid = agent_controller.mpcRM(agent, point, v_prev)
        
    v = v_rigid + [0.0] * 2
    print(v)
    agent_controller.move(agent, v, s)

    return v_rigid, finish

def approach():
    v_prev = [0.0] * 3

    while True:
        target_pose = getNearestPoint()
        v_prev, finish = goToPoint(target_pose, v_prev)
            
        if finish:
            break

def grasp():
    global agent

    v_prev = [0.0] * 2
    s = [1, 1]

    finish = False

    while True:
        target_config = getNearestPoint()
        # target_pose = target_pose[:2] + [agent.theta] + [14, 14]
        if closeToShape(agent.curvature, target_config[3:]) or rgb_camera.finish:
            v_soft = [0.0] * 2
            s = [0, 0]
            finish = True
        else:
            v_soft = agent_controller.mpcSM3(agent, target_config, v_prev)

        v = [0.0] * 3 + v_soft
        print(v)
        _, _, s_current, _ = agent_controller.move(agent, v, s)
        agent.stiffness = s_current
        
        v_prev = v_soft
            
        if finish:
            break

def transport():
    v_prev = [0.0] * 3

    target_pos = manip_target.getPoint(closest_s_index)
    orientation = manip_target.getTangent(closest_s_index)
    target_pose = target_pos + [orientation]

    while True:
        v_prev, finish = goToPoint(target_pose, v_prev)
        # print(f'Current th: {agent.theta}')
        # print(f'Target th: {orientation}')
            
        if finish:
            break

def release():
    pass

def goHome():
    pass


if __name__ == "__main__":

    # ------------------------ Start tracking -----------------------
    cheescake_contour = extractManipShape('Experiments/Data/Contours/cheescake_contour.csv')

    print('Start Motive streaming....')
    mocap.startDataListener() 
    
    update_thread = threading.Thread(target=updateConfigLoop)
    update_thread.daemon = True  
    update_thread.start()

    while not agent or not manip:
        pass

    target_manip_pose = target_manip_pos + [manip.theta-np.pi/4]
    manip_target = manipulandum.Shape(2, target_manip_pose, cheescake_contour)
    # ---------------------------------------------------------------

    # ------------------------ Start a video ------------------------
    date_title = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    rgb_camera.startVideo(date_title, task='object_handling')

    print('Waiting for the video to start...')
    while not rgb_camera.wait_video:
        pass

    print('Video started')
    print()
    # ---------------------------------------------------------------

    # ------------------------- Execute task ------------------------
    approach()
    grasp()
    transport()
    release()
    goHome()

    rgb_camera.finish = True
    # ---------------------------------------------------------------

    



        

        

            