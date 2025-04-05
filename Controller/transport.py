from Model import splines, robot2sr, manipulandum, global_var as gv
import func
import numpy as np
import time
import cvxpy
import robot2sr_controller as rsr_ctrl

def bezierCurve(t, p0, p1, p2, p3):
    return (1-t)**3 * p0 + 3*(1-t)**2 * t * p1 + 3*(1-t) * t**2 * p2 + t**3 * p3

def generatePath(manip_pose: list, manip_target_pos: list, obj_dir: float, beta=0.0) -> tuple[splines.Trajectory, float, float]:

    # Start and end points
    start = np.array(manip_pose[:-1])
    end = np.array(manip_target_pos)

    # Calculate control points for smooth exit and entrance
    exit_distance = 0.2  # Adjust this value to control the "smoothness" of the exit
    entrance_distance = 0.3 # Adjust this value to control the "smoothness" of the entrance

    p0 = start
    p1 = start + exit_distance * np.array([np.cos(obj_dir), np.sin(obj_dir)])
    p1 = start + exit_distance * np.array([np.cos(obj_dir), np.sin(obj_dir)])
    p2 = end - entrance_distance * np.array([np.cos(obj_dir + beta), 
                                             np.sin(obj_dir + beta)])
    p3 = end

    # Generate path points
    num_points = 100  # Adjust this value to control the density of points
    points = []
    for i in range(num_points):
        t = i / (num_points - 1)
        point = bezierCurve(t, p0, p1, p2, p3)
        points.append(point.tolist())

    points_array = np.array(points)
    path = splines.Trajectory(points_array[:,0], points_array[:,1])

    manip_delta_theta = path.yaw[0] - manip_pose[-1]
    manip_target_theta = path.yaw[-1] - manip_delta_theta

    return path, points, manip_delta_theta, manip_target_theta

def defineGrasp(manip: manipulandum.Shape) -> list:
    dir_angle = manip.heading_angle - np.pi
    direction_vector = np.array([np.cos(dir_angle), np.sin(dir_angle)])

    # Find a point on the contour in the opposite direction
    s_array = np.linspace(0, 1, 200)
    max_dot_product = 0
    delta = 0.02
    delta_2 = delta * 2
    delta_4 = delta * 4

    for s in s_array:
        point = manip.getPoint(s)
        theta = manip.getTangent(s)
        
        vector_to_point = np.array(point) - manip.position
        dot_product = np.dot(vector_to_point, direction_vector)
        
        if dot_product > max_dot_product:
            grasp_idx = s
            optimal_theta = func.normalizeAngle(theta)
            max_dot_product = dot_product

            final_contact_pos = [point[0] + delta * np.cos(dir_angle), 
                                 point[1] + delta * np.sin(dir_angle)]

            pre_grasp_pos = [point[0] + delta_2 * np.cos(dir_angle), 
                            point[1] + delta_2 * np.sin(dir_angle)]
            
            approach_pos = [point[0] + delta_4 * np.cos(dir_angle), 
                            point[1] + delta_4 * np.sin(dir_angle)]
            
    k1 = manip.getMeanCurvature(grasp_idx, gv.L_VSS, 'clockwise')
    k2 = manip.getMeanCurvature(grasp_idx, gv.L_VSS)

    k1 += -2 if k1 > 0 else 2
    k2 += -2 if k2 > 0 else 2

    approach_config = [*approach_pos, optimal_theta, 0, 0]
    pre_grasp_config = [*pre_grasp_pos, optimal_theta, k1, k2]
    final_config = [*final_contact_pos, optimal_theta, k1, k2]

    return grasp_idx, approach_config, pre_grasp_config, final_config

# ----------------------------- Motion Control ------------------------------

lookahead_distance = 0.04
T = 20
NX = 3
NU = 3
# R = np.diag([10000, 0.08, 0.002]) # input cost matrix (sheescake)
# R = np.diag([10000, 1.05, 0.021]) # input cost matrix (ellipse)
R = np.diag([10000, 1.05, 0.019]) # input cost matrix (heart)
# R = np.diag([10000, 1.0, 0.017]) # input cost matrix (bean)
Q = np.diag([10, 10, 0.0]) # cost matrixq
Qf = Q # final matrix
Rd = np.diag([5, 0.01, 0.001])

def arcEndPoints(agent: robot2sr.Robot, seg=1):
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

def getNearestPoint(point,  manip: manipulandum.Shape):
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
    
    # Get the closest point
    closest_point = manip_contour[closest_point_index]
    orientation = manip.getTangent(s_array[closest_point_index])
    target_pose = [closest_point[0], closest_point[1], orientation]

    return target_pose, closest_s_index

def getContactFrames(agent: robot2sr.Robot, manip: manipulandum.Shape):
    s_frames = [arcEndPoints(agent), agent.pose, arcEndPoints(agent, 2)]
    cp_object_s = [0] * 3
    c_frames = [0] * 3

    for i in range(3):
        c_frames[i], closest_s_index = getNearestPoint(s_frames[i][:-1], manip)
        cp_object_s[i] = closest_s_index

    return c_frames, closest_s_index

def calcRefTrajectory(cx: list, cy: list, cyaw: list, sp, ind, v) -> tuple[np.ndarray, np.ndarray]:
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

def getLinearModelMatrix(vref, phi):
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

def mpc(qref, vref, manip: manipulandum.Shape):
    q = cvxpy.Variable((NX, T + 1))
    u = cvxpy.Variable((NU, T))

    cost = 0.0
    constraints = []

    # constraints += [q[:, 0] == manip.pose - qref[:,0]]  
    constraints += [q[:, 0] == manip.pose_heading - qref[:,0]]  

    for t in range(T):
        cost += cvxpy.quad_form(u[:, t], R)
        if t != 0:
            cost += cvxpy.quad_form(q[:, t], Q)        
            # Add smoothness cost (penalize control input changes)
            cost += cvxpy.quad_form(u[:, t] - u[:, t-1], Rd)
        
        A, B = getLinearModelMatrix(vref[0, t], qref[2, t])  

        constraints += [q[:, t + 1] == A @ q[:, t] + B @ u[:, t]]  

    cost += cvxpy.quad_form(q[:, T], Qf)  
    prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
    prob.solve(solver=cvxpy.ECOS, verbose=False)

    if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
        vx = u.value[0, 0] + vref[0, 1]
        vy = u.value[1, 0]
        omega = u.value[2, 0]

        vel_rot = np.array([[np.cos(manip.delta_theta), -np.sin(manip.delta_theta)],
                            [np.sin(manip.delta_theta), np.cos(manip.delta_theta)]])
        v = vel_rot.dot(np.array([[vx], [vy]]))
        vx, vy = v.flatten().tolist()

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

def cpVelocities(v_o, manip: manipulandum.Shape, c_frames):
    '''
    Change Ad_oc_inv to Ad_co
    '''
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

def robotVelocities(v_c, agent: robot2sr.Robot, c_frames):
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

def transport(agent: robot2sr.Robot, manip: manipulandum.Shape, agent_controller: rsr_ctrl.Controller, manip_target_pos: list, 
              path:splines.Trajectory, rgb_camera, start_time: float, simulation=False) -> dict:
    TARGET_SPEED = 0.06

    v_r = [0.0] * 3
    v_o = [0.0] * 3
    finish = False

    tracking_data = []
    elapsed_time = 0

    cx, cy, cyaw, s = path.params
    sp = [TARGET_SPEED] * len(cx)

    rgb_camera.target_robot_config = None
    rgb_camera.traversed_trajectory = []
    rgb_camera.add2traj(manip.position)
    
    while True:
        if func.close2Pos(manip.position, manip_target_pos) or rgb_camera.finish:
            v_r = np.array([0.0] * 3)
            finish = True
        else:
            c_frames, cp_object_s = getContactFrames(agent, manip)
            target_ind = path.getTarget(manip.position, lookahead_distance)
            qref, vref = calcRefTrajectory(cx, cy, cyaw, sp, target_ind, v_o[1]) 

            v_o, q = mpc(qref, vref, manip)
            print(f'V_o: {v_o}')
            
            if simulation:
                manip.pose = q
            rgb_camera.add2traj(manip.position)

            v_c = cpVelocities(v_o, manip, c_frames)
            print(f'V_c: {v_c}')

            v_c_list = [v_c[3*i:3*i+3].tolist() for i in range(int(len(v_c)/3))]
            c_dot_new = []

            for c_i, v_c_i in zip(c_frames, v_c_list):
                J = np.array([[np.cos(c_i[-1]), -np.sin(c_i[-1]), 0],
                              [np.sin(c_i[-1]), np.cos(c_i[-1]), 0],
                              [0, 0, 1]])
                
                c_i_dot = J.dot(v_c_i)
                c_dot_new.append(c_i_dot.tolist())

            v_r = robotVelocities(v_c, agent, c_frames)
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
        _, agent_q_new, _, _ = agent_controller.move(agent, v, [0, 0])
        print()

        if simulation:
            agent.config = agent_q_new

        if finish:
            break

    print()
    print(f'Recording time: {elapsed_time} seconds')

    path_data = []
    for x, y, yaw, in zip(path.x, path.y, path.yaw):
        path_data.append({'x': x, 'y': y, 'yaw': yaw})

    # Prepare data to be written
    transport_data = {
        'path': path_data,
        "tracking": tracking_data
    }

    return transport_data

