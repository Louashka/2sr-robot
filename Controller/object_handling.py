# Make grasping first:
# 1. Select the closest point on the object's contour
# 2. Approach the point (misth some pre margin) in a rigid mode
# 3. Grasp the object in a soft mode SM3

import sys
sys.path.append('D:/Robot 2SR/2sr-swarm-control')
from Model import global_var as gv, robot2sr as rsr, manipulandum, splines
import motive_client, robot2sr_controller as rsr_ctrl, camera_optitrack_synchronizer as cos
import pandas as pd
import threading
from datetime import datetime
import numpy as np
import pandas as pd
import time
import json
import cvxpy

mocap = motive_client.MocapReader()
rgb_camera = cos.Aligner()

markers = None

agent: rsr.Robot = None
agent_controller = rsr_ctrl.Controller()

contour = None
manip: manipulandum.Shape = None
manip_target: manipulandum.Shape = None

cheescake_targets = [[[0.270,  0.826], -np.pi/2],
                     [[-0.385, 0.723], np.pi/3],
                     [[0.149,  0.840], -np.pi/6]]
target_i = 2

# target_manip_pos = [0.25, 1.04]
# target_manip_pos = [-0.1345, 1.04]
# target_manip_pos = [0.27, 0.826]
# target_manip_pos = [-0.385, 0.723]
# target_manip_pos = [0.149, 0.84]
# target_manip_pos = [-0.415, -0.3]
traj: splines.Trajectory = None
closest_s_index = None

s_frames = []
c_frames = []

c_dot = []
frames_index = []
cp_object_s = [0] * 3

simulation = True
# simulation = False

T = 20
NX = 3
NU = 3

# mpc parameters
R = np.diag([10000, 1, 0.0028])  # input cost matrix
Q = np.diag([10, 10, 0.0])  # cost matrix
Qf = Q # final matrix
Rd = np.diag([10, 10000, 0.001])

sp = None
TARGET_SPEED = 0.053
lookahead_distance = 0.04

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
            if not simulation:
                # manip_pose[0] += manip.r * np.cos(manip.theta + manip.phi)
                # manip_pose[1] += manip.r * np.sin(manip.theta + manip.phi)
                manip.pose = manip_pose
            else:
                pass
        else:
            manip = manipulandum.Shape(manip_config['id'], manip_pose, contour)
        # print(f'Manip\'s pose: {manip.pose}')
    else:
        print('Agent and/or manipulandum is not detected! ' + msg)

def updateConfigLoop():
    global rgb_camera, agent, markers, s_frames, c_frames, frames_index, cp_object_s

    while True:
        updateConfig()
        if rgb_camera is not None and agent is not None and manip_target is not None:
            rgb_camera.markers = markers
            rgb_camera.manip_center = manip.pose
            rgb_camera.manip_target_contour = manip_target.contour
        
            s_frames = [arc_endpoints(), agent.pose, arc_endpoints(2)]

            if simulation:
                if len(frames_index) == 3:
                    c_frames = [getPoint(index) for index in frames_index]
                else:
                    c_frames = [getNearestPoint(s_frame[:-1]) for s_frame in s_frames]
            else:
                frames = [0] * 3
                for i in range(3):
                    frames[i] = getNearestPoint(s_frames[i][:-1])
                    cp_object_s[i] = closest_s_index
                c_frames = frames
            
            # rgb_camera.contact_points = c_frames

            # rgb_camera.c_dot = c_dot

def arc_endpoints(seg=1):
    global agent

    if seg == 1:
        flag = -1
        k = agent.k1
    else:
        flag = 1
        k = agent.k2

    theta = agent.theta + flag * k * gv.L_VSS

    if k == 0:
        x = flag * gv.L_VSS * np.cos(agent.theta)
        y = flag * gv.L_VSS * np.sin(agent.theta)
    else:
        x = np.sin(theta) / k - np.sin(agent.theta) / k
        y = -np.cos(theta) / k + np.cos(agent.theta) / k

    x += agent.x
    y += agent.y
        
    return [x, y, theta]

def getNearestPoint(point):
    global closest_s_index, frames_index
    # Find the closest point on manip.contour to agent.position
    pos = np.array(point)  # Only consider x and y coordinates
    # manip_contour = manip.parametric_contour.T  # Transpose to get a list of [x, y] points
    s_array, manip_contour = manip.parametric_contour
    manip_contour = manip_contour.T
    
    # Calculate distances from agent to all points on the contour
    distances = np.linalg.norm(manip_contour - pos, axis=1)
    
    # Find the index of the minimum distance
    closest_point_index = np.argmin(distances)
    closest_s_index = s_array[closest_point_index]

    if len(frames_index) < 3:
        frames_index.append(closest_point_index)
    
    # Get the closest point
    closest_point = manip_contour[closest_point_index]
    orientation = manip.getTangent(s_array[closest_point_index])
    target_pose = [closest_point[0], closest_point[1], orientation]

    return target_pose

def getPoint(index):
    s_array, manip_contour = manip.parametric_contour
    manip_contour = manip_contour.T

    pos = manip_contour[index]
    orientation = manip.getTangent(s_array[index])
    point = [pos[0], pos[1], orientation]

    return point
    
def updateContour():
    global rgb_camera, manip
    # Convert cheescake_contour to phase angles and radiuses with respect to manip.pose
    if rgb_camera.detected_object_contour is not None and manip is not None:
        manip_x, manip_y, manip_theta = manip.pose
        phase_angles = []
        radiuses = []
        
        for point in rgb_camera.detected_object_contour:
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

        # Save phase angles, radiuses, and manip pose to CSV file
        csv_file_path = 'Experiments/Data/Contours/bean_contour.csv'
        # Create a DataFrame from the phase angles, radiuses, and manip pose
        df = pd.DataFrame({'phase_angle': phase_angles, 'radius': radiuses})

        # Save the DataFrame to a CSV file
        df.to_csv(csv_file_path, index=False)
        print(f"Object contour data saved to {csv_file_path}")

def closeToGoal(current, target):
    status = True

    # Calculate Euclidean distance between current and target (x, y)
    distance = np.linalg.norm(np.array(current[:2]) - np.array(target[:2]))
    
    # Calculate absolute difference between current and target theta
    theta_difference = abs(current[2] - target[2])
    
    # Define thresholds for position and orientation
    distance_threshold = 0.011  
    theta_threshold = 0.1  
    
    # Check if both position and orientation are within thresholds
    # if distance > distance_threshold or theta_difference > theta_threshold:
    #     status = False

    if distance > distance_threshold:
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

def bezier_curve(t, p0, p1, p2, p3):
        return (1-t)**3 * p0 + 3*(1-t)**2 * t * p1 + 3*(1-t) * t**2 * p2 + t**3 * p3

def generatePath():
    global manip, manip_target, traj

    # Start and end points
    start = np.array(manip.position)
    end = np.array(manip_target.position)

    # Calculate control points for smooth exit and entrance
    exit_distance = 0.85  # Adjust this value to control the "smoothness" of the exit
    entrance_distance = 0.85  # Adjust this value to control the "smoothness" of the entrance

    p0 = start
    p1 = start + exit_distance * np.array([np.cos(manip.theta), np.sin(manip.theta)])
    p2 = end - entrance_distance * np.array([np.cos(manip_target.theta + cheescake_targets[target_i][1]), 
                                             np.sin(manip_target.theta + cheescake_targets[target_i][1])])
    p3 = end

    # Generate path points
    num_points = 100  # Adjust this value to control the density of points
    points = []
    for i in range(num_points):
        t = i / (num_points - 1)
        point = bezier_curve(t, p0, p1, p2, p3)
        points.append(point.tolist())

    path = np.array(points)
    traj = splines.Trajectory(path[:,0], path[:,1])

    return points

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

def transport(date_title):
    global c_frames, c_dot, sp

    v_r = [0.0] * 3
    v_o = [0.0] * 3
    finish = False
# 
    rgb_camera.add_to_traj(manip.position)

    directory = "Experiments/Data/Tracking/Object_transport"
    filename = f"{directory}/heart_{date_title}.json"

    tracking_data = []
    elapsed_time = 0
    start_time = time.perf_counter()
    
    while True:
        if closeToGoal(manip.pose, manip_target.pose) or rgb_camera.finish:
            v_r = np.array([0.0] * 3)
            finish = True
        else:
            # v_o, q = simple_control(manip.pose, manip_target.pose)

            cx, cy, cyaw, s = traj.params

            if sp is None:
                sp = []
                for i in range(1, len(cx)):
                    sp.append(TARGET_SPEED * (1 - traj.curvature[i]/(max(traj.curvature) + 6)))
                sp.append(2 * sp[-1]/3)

            target_ind = traj.getTarget(manip.position, lookahead_distance)
            qref, vref = calc_ref_trajectory(cx, cy, cyaw, sp, target_ind, v_o[1]) 

            v_o, q = mpc(qref, vref)
            print(f'V_o: {v_o}')
            
            if simulation:
                manip.pose = q
            rgb_camera.add_to_traj(manip.position)

            v_c = cp_velocities(v_o)
            print(f'V_c: {v_c}')

            v_c_list = [v_c[3*i:3*i+3].tolist() for i in range(int(len(v_c)/3))]
            c_dot_new = []

            for c_i, v_c_i in zip(c_frames, v_c_list):
                J = np.array([[np.cos(c_i[-1]), -np.sin(c_i[-1]), 0],
                              [np.sin(c_i[-1]), np.cos(c_i[-1]), 0],
                              [0, 0, 1]])
                
                c_i_dot = J.dot(v_c_i)
                c_dot_new.append(c_i_dot.tolist())

            c_dot = c_dot_new

            v_r = robot_velocities(v_c)
            print(f'V_r: {v_r}')

            current_time = time.perf_counter()
            elapsed_time = current_time - start_time

            object_data = {'pose' : manip.pose,
                           'target_velocity': v_o}
            robot_data = {'pose' : agent.pose,
                          'target_velocity': v_r.tolist()}

            tracking_data.append({'time': elapsed_time,
                                  'object': object_data,
                                  'robot': robot_data,
                                  'cp': cp_object_s})

        v = v_r.tolist() + [0.0] * 2
        agent_controller.move(agent, v, [0, 0])

        if finish:
            break

    print()
    print(f'Recording time: {elapsed_time} seconds')

    path_data = []
    for x, y, yaw, in zip(traj.x, traj.y, traj.yaw):
        path_data.append({'x': x, 'y': y, 'yaw': yaw})

    # Prepare data to be written
    data_json = {
        "metadata": {
            "description": "Object manipulation",
            "date": date_title
        },
        'path': path_data,
        "tracking": tracking_data
    }

    # Write data to JSON file
    if not simulation:
        with open(filename, 'w') as f:
            json.dump(data_json, f, indent=2)

        print(f"Data written to {filename}")

def map_to_pi_range(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi
    
def simple_control(current, target):
    q = np.array(current)
    q_t = np.array(target)    

    q[-1] = map_to_pi_range(q[-1])
    q_t[-1] = map_to_pi_range(q_t[-1])

    print(f'Current angle: {q[-1]}')
    print(f'Target angle: {q_t[-1]}')

    coef = 0.075
    q_tilda = (q_t - q) * coef

    rot = np.array([[np.cos(q[2]), -np.sin(q[2]), 0],
                    [np.sin(q[2]), np.cos(q[2]), 0],
                    [0, 0, 1]])
    
    v = np.linalg.inv(rot).dot(q_tilda)

    q_dot = rot.dot(v)
    q_new = q + q_dot * gv.DT

    return v, q_new

def cp_velocities(v_o):
    global c_frames

    rot_ow = np.array([[np.cos(-manip.theta), -np.sin(-manip.theta)],
                       [np.sin(-manip.theta), np.cos(-manip.theta)]])
    # B_c = np.array([[1, 0], [0, 1], [0, 0]])
    B_c = np.identity(3)
    G_T_list = []
    v_c = None

    for c_i in c_frames:
        dist = np.array([c_i[0] - manip.x, c_i[1] - manip.y]).reshape(2,1)
        
        pos = rot_ow.dot(dist)
        theta = c_i[2] - manip.theta

        rot_oc = np.array([[np.cos(theta), -np.sin(theta)],
                           [np.sin(theta), np.cos(theta)]])
        
        Ad_oc_inv = np.block([[rot_oc.T, rot_oc.T.dot(np.array([[-pos[1,0]], [pos[0,0]]]))], 
                              [np.zeros((1, 2)), 1]])
        
        G_T_i = B_c.T.dot(Ad_oc_inv)
        G_T_list.append(G_T_i)

    if G_T_list:
        G_T = np.vstack(G_T_list)
        v_c = G_T.dot(v_o)

    return v_c

def robot_velocities(v_c):
    global c_frames

    rot_rw = np.array([[np.cos(-agent.theta), -np.sin(-agent.theta)],
                       [np.sin(-agent.theta), np.cos(-agent.theta)]])
    # B_c = np.array([[1, 0], [0, 1], [0, 0]])
    B_c = np.identity(3)
    angles_offset = [0, -np.pi/2, np.pi/2]
    G_list = []
    v_r = None

    for c_i, angle_offset in zip(c_frames, angles_offset):
        dist = np.array([c_i[0] - agent.x, c_i[1] - agent.y]).reshape(2,1)
        
        pos = rot_rw.dot(dist)
        theta = c_i[2] - agent.theta + angle_offset

        rot_rc = np.array([[np.cos(theta), -np.sin(theta)],
                           [np.sin(theta), np.cos(theta)]])
        
        Ad_rc_inv_T = np.block([[rot_rc, np.zeros((2, 1))], 
                                [np.array([[-pos[1,0], pos[0,0]]]).dot(rot_rc), 1]])
        
        G_i = Ad_rc_inv_T.dot(B_c)
        G_list.append(G_i)

    if G_list:
        G = np.block(G_list)
        v_r = G.dot(v_c)

    return v_r

def calc_ref_trajectory(cx: list, cy: list, cyaw: list, sp, ind, v) -> tuple[np.ndarray, np.ndarray]:
    qref = np.zeros((NX, T + 1))
    vref = np.zeros((1, T + 1))
    ncourse = len(cx)

    qref[0, 0] = cx[ind]
    qref[1, 0] = cy[ind]
    qref[2, 0] = cyaw[ind]
    vref[0, 0] = sp[ind]
    travel = 0.0

    for i in range(1, T + 1):
        travel += abs(v) * gv.DT
        dind = int(round(travel / lookahead_distance))
        if (ind + dind) < ncourse:
            qref[0, i] = cx[ind + dind]
            qref[1, i] = cy[ind + dind]
            qref[2, i] = cyaw[ind + dind]
            vref[0, i] = sp[ind + dind]
        else:
            qref[0, i] = cx[ncourse - 1]
            qref[1, i] = cy[ncourse - 1]
            qref[2, i] = cyaw[ncourse - 1]
            vref[0, i] = sp[ncourse - 1]

    return qref, vref

def mpc(qref, vref):
    q = cvxpy.Variable((NX, T + 1))
    u = cvxpy.Variable((NU, T))

    cost = 0.0
    constraints = []

    constraints += [q[:, 0] == manip.pose - qref[:,0]]  

    for t in range(T):
        cost += cvxpy.quad_form(u[:, t], R)
        if t != 0:
            cost += cvxpy.quad_form(q[:, t], Q)        
        A, B = get_linear_model_matrix(vref[0, t], qref[2, t])  

        constraints += [q[:, t + 1] == A @ q[:, t] + B @ u[:, t]]  

    cost += cvxpy.quad_form(q[:, T], Qf)  
    prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
    prob.solve(solver=cvxpy.ECOS_BB, verbose=False)

    if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
        vx = u.value[0, 0] + vref[0, 1]
        vy = u.value[1, 0]
        omega = u.value[2, 0]

        rot = np.array([[np.cos(manip.theta), -np.sin(manip.theta), 0],
                        [np.sin(manip.theta), np.cos(manip.theta), 0],
                        [0, 0, 1]])

        q_dot = rot.dot([vx, vy, omega])
        q_new = np.array(manip.pose) + q_dot * gv.DT
    else:
        print("Error: Cannot solve mpc..")
        vx, vy, omega = None, None, None
        q_new = None

    return [vx, vy, omega], q_new

def get_linear_model_matrix(vref, phi):
    A = np.zeros((NX, NX))
    A[0, 0] = 1.0
    A[0, 2] = -vref * np.sin(phi) * gv.DT
    A[1, 1] = 1.0
    A[1, 2] = vref * np.cos(phi) * gv.DT
    A[2, 2] = 1.0

    B = np.zeros((NX, NU))
    B[0, 0] = gv.DT * np.cos(phi)
    B[0, 1] = -gv.DT * np.sin(phi)
    B[1, 0] = gv.DT * np.sin(phi)
    B[1, 1] = gv.DT * np.cos(phi)
    B[2, 2] = gv.DT

    return A, B

def release():
    pass

def goHome():
    pass


if __name__ == "__main__":

    # ------------------------ Start tracking -----------------------
    contour = extractManipShape('Experiments/Data/Contours/bean_contour.csv')

    print('Start Motive streaming....')
    mocap.startDataListener() 
    
    update_thread = threading.Thread(target=updateConfigLoop)
    update_thread.daemon = True  
    update_thread.start()

    while not agent or not manip:
        pass

    target_manip_pose = cheescake_targets[target_i][0] + [manip.theta]
    manip_target = manipulandum.Shape(2, target_manip_pose, contour)
    # ---------------------------------------------------------------

    rgb_camera.path = generatePath()
    manip_target.theta = traj.yaw[-1]

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
    # approach()
    # grasp()
    # transport(date_title)
    rgb_camera.contour = manip.contour
    release()
    goHome()
    # updateContour()

    rgb_camera.finish = True
    # ---------------------------------------------------------------

    



        

        

            