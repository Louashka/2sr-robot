import sys
from Motive import nat_net_client as nnc, mo_cap_data
from Model import global_var
import numpy as np
import pandas as pd
from typing import List
from scipy.optimize import minimize 
from scipy.interpolate import splprep, splev
from circle_fit import taubinSVD
from scipy.signal import savgol_filter, butter, filtfilt
from collections import deque
import matplotlib.pyplot as plt

m_pos = ['marker_x', 'marker_y', 'marker_z']
pos_2d = ['marker_x', 'marker_y']
rb_pos = ['x', 'y', 'z']
rb_params = ['a', 'b', 'c', 'd']
rb_angles = ['roll', 'pitch', 'yaw']

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['text.usetex'] = False
plt.rcParams['font.size'] = 12

class MotionDataFilter:
    def __init__(self, pose_window_size=11, curvature_window_size=15, history_length=120):
        """Initialize the filter with parameters.
        
        Args:
            window_size: Size of the Savitzky-Golay filter window
            history_length: Number of frames to keep in history for filtering
        """
        self.pose_window_size = pose_window_size
        self.curvature_window_size = curvature_window_size
        self.history_length = history_length
        
        # Data history storage
        self.marker_history = {}       # Store marker position history
        self.rigid_body_history = {}   # Store rigid body position/orientation history
        self.agent_config_history = {}       # Store processed configuration history
        self.object_config_history = {}
        
    def update_marker_history(self, markers):
        """Update the history of marker positions."""
        for marker_id, marker_data in markers.items():
            if marker_id not in self.marker_history:
                self.marker_history[marker_id] = {
                    'x': deque(maxlen=self.history_length),
                    'y': deque(maxlen=self.history_length),
                    'z': deque(maxlen=self.history_length)
                }
            
            # Add current position to history
            self.marker_history[marker_id]['x'].append(marker_data['marker_x'])
            self.marker_history[marker_id]['y'].append(marker_data['marker_y'])
            self.marker_history[marker_id]['z'].append(marker_data['marker_z'])
    
    def update_rigid_body_history(self, rigid_bodies):
        """Update the history of rigid body positions and orientations."""
        for rb_id, rb_data in rigid_bodies.items():
            if rb_id not in self.rigid_body_history:
                self.rigid_body_history[rb_id] = {
                    'x': deque(maxlen=self.history_length),
                    'y': deque(maxlen=self.history_length),
                    'z': deque(maxlen=self.history_length),
                    'a': deque(maxlen=self.history_length),
                    'b': deque(maxlen=self.history_length),
                    'c': deque(maxlen=self.history_length),
                    'd': deque(maxlen=self.history_length)
                }
            
            # Add current position to history
            self.rigid_body_history[rb_id]['x'].append(rb_data['x'])
            self.rigid_body_history[rb_id]['y'].append(rb_data['y'])
            self.rigid_body_history[rb_id]['z'].append(rb_data['z'])
            self.rigid_body_history[rb_id]['a'].append(rb_data['a'])
            self.rigid_body_history[rb_id]['b'].append(rb_data['b'])
            self.rigid_body_history[rb_id]['c'].append(rb_data['c'])
            self.rigid_body_history[rb_id]['d'].append(rb_data['d'])
    
    def update_agent_config_history(self, agent_id, config):
        """Update the history of robot configuration values."""
        if agent_id not in self.agent_config_history:
            self.agent_config_history[agent_id] = {
                'x': deque(maxlen=self.history_length),
                'y': deque(maxlen=self.history_length),
                'theta': deque(maxlen=self.history_length),
                'k1': deque(maxlen=self.history_length),
                'k2': deque(maxlen=self.history_length)
            }
        
        # Add current configuration to history
        self.agent_config_history[agent_id]['x'].append(config['x'])
        self.agent_config_history[agent_id]['y'].append(config['y'])
        self.agent_config_history[agent_id]['theta'].append(config['theta'])
        self.agent_config_history[agent_id]['k1'].append(config['k1'])
        self.agent_config_history[agent_id]['k2'].append(config['k2'])
    
    def update_object_config_history(self, object_id, config):
        """Update the history of robot configuration values."""
        if object_id not in self.object_config_history:
            self.object_config_history[object_id] = {
                'x': deque(maxlen=self.history_length),
                'y': deque(maxlen=self.history_length),
                'theta': deque(maxlen=self.history_length)
            }
        
        # Add current configuration to history
        self.object_config_history[object_id]['x'].append(config['x'])
        self.object_config_history[object_id]['y'].append(config['y'])
        self.object_config_history[object_id]['theta'].append(config['theta'])
    
    def filter_markers(self, markers):
        """Apply filtering to marker positions."""
        filtered_markers = {}
        
        for marker_id, marker_data in markers.items():
            # First update history
            if marker_id in self.marker_history:
                # We need enough data points for filtering
                x_history = list(self.marker_history[marker_id]['x'])
                y_history = list(self.marker_history[marker_id]['y'])
                z_history = list(self.marker_history[marker_id]['z'])
                
                if len(x_history) >= self.pose_window_size:
                    # Apply Savitzky-Golay filter
                    win_size = min(self.pose_window_size, len(x_history) - (len(x_history) % 2) - 1)
                    if win_size >= 3:  # Need at least 3 points for quadratic fit
                        x_filtered = savgol_filter(x_history, win_size, 2)[-1]
                        y_filtered = savgol_filter(y_history, win_size, 2)[-1]
                        z_filtered = savgol_filter(z_history, win_size, 2)[-1]
                        
                        # Create a filtered marker
                        filtered_marker = marker_data.copy()
                        filtered_marker['marker_x'] = x_filtered
                        filtered_marker['marker_y'] = y_filtered
                        filtered_marker['marker_z'] = z_filtered
                        
                        filtered_markers[marker_id] = filtered_marker
                    else:
                        filtered_markers[marker_id] = marker_data
                else:
                    filtered_markers[marker_id] = marker_data
            else:
                filtered_markers[marker_id] = marker_data
        
        return filtered_markers
    
    def filter_rigid_bodies(self, rigid_bodies):
        """Apply filtering to rigid body positions and orientations."""
        filtered_rigid_bodies = {}
        
        for rb_id, rb_data in rigid_bodies.items():
            if rb_id in self.rigid_body_history:
                # Get history for this rigid body
                x_history = list(self.rigid_body_history[rb_id]['x'])
                y_history = list(self.rigid_body_history[rb_id]['y'])
                z_history = list(self.rigid_body_history[rb_id]['z'])
                
                # For quaternions, we need special handling
                a_history = list(self.rigid_body_history[rb_id]['a'])
                b_history = list(self.rigid_body_history[rb_id]['b'])
                c_history = list(self.rigid_body_history[rb_id]['c'])
                d_history = list(self.rigid_body_history[rb_id]['d'])
                
                if len(x_history) >= self.pose_window_size:
                    # Apply Savitzky-Golay filter to position
                    win_size = min(self.pose_window_size, len(x_history) - (len(x_history) % 2) - 1)
                    if win_size >= 3:
                        x_filtered = savgol_filter(x_history, win_size, 2)[-1]
                        y_filtered = savgol_filter(y_history, win_size, 2)[-1]
                        z_filtered = savgol_filter(z_history, win_size, 2)[-1]
                        
                        # Filter quaternions - important to treat as unit quaternions
                        # For simplicity here just filtering components, but more sophisticated
                        # approaches exist for quaternion filtering
                        a_filtered = savgol_filter(a_history, win_size, 2)[-1]
                        b_filtered = savgol_filter(b_history, win_size, 2)[-1]
                        c_filtered = savgol_filter(c_history, win_size, 2)[-1]
                        d_filtered = savgol_filter(d_history, win_size, 2)[-1]
                        
                        # Normalize the filtered quaternion
                        q_norm = np.sqrt(a_filtered**2 + b_filtered**2 + c_filtered**2 + d_filtered**2)
                        a_filtered /= q_norm
                        b_filtered /= q_norm
                        c_filtered /= q_norm
                        d_filtered /= q_norm
                        
                        # Create filtered rigid body
                        filtered_rb = rb_data.copy()
                        filtered_rb['x'] = x_filtered
                        filtered_rb['y'] = y_filtered
                        filtered_rb['z'] = z_filtered
                        filtered_rb['a'] = a_filtered
                        filtered_rb['b'] = b_filtered
                        filtered_rb['c'] = c_filtered
                        filtered_rb['d'] = d_filtered
                        
                        filtered_rigid_bodies[rb_id] = filtered_rb
                    else:
                        filtered_rigid_bodies[rb_id] = rb_data
                else:
                    filtered_rigid_bodies[rb_id] = rb_data
            else:
                filtered_rigid_bodies[rb_id] = rb_data
        
        return filtered_rigid_bodies
    
    def filter_pose(self, id_, config, entity='robot'):
        filtered_config = config.copy()

        if entity == 'robot':
            history = self.agent_config_history
        elif entity == 'object': 
            history = self.object_config_history

        if id_ in history:
            # Get history for this agent
            x_history = list(history[id_]['x'])
            y_history = list(history[id_]['y'])
            theta_history = list(history[id_]['theta'])

            if len(x_history) >= self.pose_window_size:
                pose_win_size = min(self.pose_window_size, len(x_history) - (len(x_history) % 2) - 1)
                if pose_win_size >= 3:
                    # Filter position and theta (angle needs special handling)
                    filtered_config['x'] = savgol_filter(x_history, pose_win_size, 2)[-1]
                    filtered_config['y'] = savgol_filter(y_history, pose_win_size, 2)[-1]
                    
                    # For theta (angular value), we need to handle circular values
                    # Convert to complex numbers, filter, then convert back to angle
                    complex_angles = np.exp(1j * np.array(theta_history))
                    real_part = savgol_filter(np.real(complex_angles), pose_win_size, 2)[-1]
                    imag_part = savgol_filter(np.imag(complex_angles), pose_win_size, 2)[-1]
                    filtered_config['theta'] = np.angle(complex(real_part, imag_part))

        return filtered_config
    
    def filter_config(self, agent_id, config):
        filtered_config = self.filter_pose(agent_id, config)
        
        if agent_id in self.agent_config_history:
            k1_history = list(self.agent_config_history[agent_id]['k1'])
            k2_history = list(self.agent_config_history[agent_id]['k2'])
            
            if len(k1_history) >= self.curvature_window_size:
                curv_win_size = min(self.curvature_window_size, len(k1_history) - (len(k1_history) % 2) - 1)
                if curv_win_size >= 3:
                    # Filter curvatures with stronger filtering (higher noise expected)
                    filtered_config['k1'] = savgol_filter(k1_history, curv_win_size, 2)[-1]
                    filtered_config['k2'] = savgol_filter(k2_history, curv_win_size, 2)[-1]
        
        return filtered_config

    def calculate_dimensionless_jerk(self, agent_id, window=30):
        """Calculate dimensionless jerk before and after filtering."""
        if agent_id not in self.agent_config_history:
            return None, None
        
        if len(self.agent_config_history[agent_id]['x']) < window:
            return None, None
        
        # Get position histories
        x_raw = list(self.agent_config_history[agent_id]['x'])[-window:]
        y_raw = list(self.agent_config_history[agent_id]['y'])[-window:]
        
        # Get filtered positions
        x_filtered = savgol_filter(x_raw, min(9, window - (window % 2) - 1), 2)
        y_filtered = savgol_filter(y_raw, min(9, window - (window % 2) - 1), 2)
        
        # Assume constant time steps of 1 for simplicity
        time_steps = np.arange(len(x_raw))
        
        # Calculate velocities
        vx_raw = np.gradient(x_raw, time_steps)
        vy_raw = np.gradient(y_raw, time_steps)
        
        vx_filtered = np.gradient(x_filtered, time_steps)
        vy_filtered = np.gradient(y_filtered, time_steps)
        
        # Calculate accelerations
        ax_raw = np.gradient(vx_raw, time_steps)
        ay_raw = np.gradient(vy_raw, time_steps)
        
        ax_filtered = np.gradient(vx_filtered, time_steps)
        ay_filtered = np.gradient(vy_filtered, time_steps)
        
        # Calculate jerks
        jx_raw = np.gradient(ax_raw, time_steps)
        jy_raw = np.gradient(ay_raw, time_steps)
        
        jx_filtered = np.gradient(ax_filtered, time_steps)
        jy_filtered = np.gradient(ay_filtered, time_steps)
        
        # Calculate path length
        path_raw = np.sum(np.sqrt(np.diff(x_raw)**2 + np.diff(y_raw)**2))
        path_filtered = np.sum(np.sqrt(np.diff(x_filtered)**2 + np.diff(y_filtered)**2))
        
        # Calculate movement duration
        duration = time_steps[-1] - time_steps[0]
        
        # Calculate squared jerk
        squared_jerk_raw = np.sum(jx_raw**2 + jy_raw**2)
        squared_jerk_filtered = np.sum(jx_filtered**2 + jy_filtered**2)
        
        # Calculate dimensionless jerk
        dj_raw = (duration**5 / path_raw**2) * squared_jerk_raw
        dj_filtered = (duration**5 / path_filtered**2) * squared_jerk_filtered
        
        return dj_raw, dj_filtered

    def create_window_evaluation_plot(self, agent_id):
        # Get data from history
        x_raw = list(self.agent_config_history[agent_id]['x'])[-100:]
        y_raw = list(self.agent_config_history[agent_id]['y'])[-100:]
        th_raw = list(self.agent_config_history[agent_id]['theta'])[-100:]
        k1_raw = list(self.agent_config_history[agent_id]['k1'])[-100:]
        k2_raw = list(self.agent_config_history[agent_id]['k2'])[-100:]
            
        plt.figure(figsize=(15, 10))
        
        # Position plot
        plt.subplot(2, 2, 1)
        x_filtered = savgol_filter(x_raw, self.pose_window_size, 2)
        y_filtered = savgol_filter(y_raw, self.pose_window_size, 2)
        
        plt.plot(x_raw, y_raw, 'r-', alpha=0.5, label='Raw')
        plt.plot(x_filtered, y_filtered, 'b-', label='Filtered')
        plt.title(f'Position (Window={self.pose_window_size})')
        plt.legend()

        # Orientation plot
        plt.subplot(2, 2, 2)
        th_filtered = savgol_filter(th_raw, self.pose_window_size, 2)
        plt.plot(th_raw, 'r-', alpha=0.5, label='Raw')
        plt.plot(th_filtered, 'b-', label='Filtered')
        plt.title(f'Orientation (Window={self.pose_window_size})')
        plt.legend()
        
        # Curvature plots
        plt.subplot(2, 2, 3)
        k1_filtered = savgol_filter(k1_raw, self.curvature_window_size, 2)
        plt.plot(k1_raw, 'r-', alpha=0.5, label='Raw')
        plt.plot(k1_filtered, 'b-', label='Filtered')
        plt.title(f'Curvature k1 (Window={self.curvature_window_size})')
        plt.legend()
        
        plt.subplot(2, 2, 4)
        k2_filtered = savgol_filter(k2_raw, self.curvature_window_size, 2)
        plt.plot(k2_raw, 'r-', alpha=0.5, label='Raw')
        plt.plot(k2_filtered, 'b-', label='Filtered')
        plt.title(f'Curvature k2 (Window={self.curvature_window_size})')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('signal_filtering.pdf', format='pdf', dpi=150, bbox_inches='tight')
        plt.show()

class Marker():
    def __init__(self, marker_id: int, x: float, y: float) -> None:
        self.__marker_id = marker_id
        self.x = x
        self.y = y

    @property
    def marker_id(self) -> int:
        return self.__marker_id

    @property
    def x(self) -> float:
        return self.__x
    
    @x.setter
    def x(self, value: float) -> None:
        self.__x = value

    @property
    def y(self) -> float:
        return self.__y
    
    @y.setter
    def y(self, value: float) -> None:
        self.__y = value

    @property
    def position(self) -> list:
        return [self.x, self.y]

class MocapReader:
    def __init__(self) -> None:
        self.__isRunning = False
        self.__data = None
        self.__markers_id = set()
        self.agent_theta_prev = None

        self.filter = MotionDataFilter()

    @property
    def isRunning(self) -> bool:
        return self.__isRunning
    
    @isRunning.setter
    def isRunning(self, value: bool) -> None:
        self.__isRunning = value

    @property
    def data(self) -> mo_cap_data.MoCapData:
        return self.__data

    def __receiveData(self, value) -> None:
        self.__data = value

    def __parseArgs(self, arg_list, args_dict) -> dict:
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
    
    def startDataListener(self) -> None:
        options_dict = {}
        options_dict["clientAddress"] = "127.0.0.1"
        options_dict["serverAddress"] = "127.0.0.1"
        options_dict["use_multicast"] = True

        # Start Motive streaning
        options_dict = self.__parseArgs(sys.argv, options_dict)

        streaming_client = nnc.NatNetClient()
        streaming_client.set_client_address(options_dict["clientAddress"])
        streaming_client.set_server_address(options_dict["serverAddress"])
        streaming_client.set_use_multicast(options_dict["use_multicast"])

        streaming_client.mocap_data_listener = self.__receiveData

        self.isRunning = streaming_client.run()

    """
    Determine the current configuration of the single 2SR agent
    :return: dict, agent's generalized coordinates; dict, markers' id and coordinates
    """

    def getConfig(self) -> tuple[list, list, dict, str]:
        agents = []
        objects = []

        # Parse Motive data
        markers, rigid_bodies = self.__unpackData()
        # markers, rigid_bodies = self.__simulateData()

        if markers is None or rigid_bodies is None:
            msg = 'No data is received from Motive!'
        
        elif not markers or not rigid_bodies:
            msg = 'No markers or rigid bodies are detected!'
        else:
            self.filter.update_marker_history(markers)
            self.filter.update_rigid_body_history(rigid_bodies)

            if len(list(self.filter.marker_history.values())[0]['x']) >= 5:
                filtered_markers = self.filter.filter_markers(markers)
                filtered_rigid_bodies = self.filter.filter_rigid_bodies(rigid_bodies)
            else:
                filtered_markers = markers
                filtered_rigid_bodies = rigid_bodies

            # Convert values from the Motive frame to the global frame
            self.__convertData(filtered_rigid_bodies)

            rb_agents = [rb for rb in filtered_rigid_bodies.values() if rb['id'] < 10]
            rb_objects = [rb for rb in filtered_rigid_bodies.values() if rb['id'] > 10]

            if len(filtered_markers) == 9 * len(rb_agents) + 3 * len(rb_objects):
                for rb_agent in rb_agents:
                    agent = {}

                    agent['id'] = rb_agent['id']
                    # Calculate the pose of the head LU
                    head_markers = [marker for marker in filtered_markers.values() if marker['model_id'] == rb_agent['id']]
                    head_pose = self.__calcRbPose(rb_agent, head_markers)
                    agent['head'] = head_pose

                    # Sort markers to within the VSF
                    ranked_markers = self.__rankPoints(filtered_markers, head_pose[:-1])
                    # Calculate the pose of the tail LU
                    tail_theta = self.__getAngle(ranked_markers[-2].position, ranked_markers[-1].position) + 0.31615
                    tail_pose = [ranked_markers[-1].x, ranked_markers[-1].y, tail_theta]
                    agent['tail'] = tail_pose
                    
                    # Calculate the robot's configuration
                    robot_x, robot_y, robot_theta, k1, k2 = self.__extrapolateCurve(ranked_markers[:-1])

                    agent['x'] = robot_x
                    agent['y'] = robot_y
                    agent['theta'] = robot_theta
                    agent['k1'] = k1
                    agent['k2'] = k2

                    self.filter.update_agent_config_history(agent['id'], agent)
                    if len(list(self.filter.agent_config_history.get(agent['id'], {'x': []})['x'])) >= 5:
                        agent = self.filter.filter_config(agent['id'], agent)

                    agents.append(agent)

                for rb_object in rb_objects:
                    object = {}

                    object['id'] = rb_object['id']

                    object_markers = [marker for marker in markers.values() if marker['model_id'] == rb_object['id']]                
                    object_pose = self.__calcRbPose(rb_object, object_markers)

                    object['x'] = object_pose[0]
                    object['y'] = object_pose[1]
                    object['theta'] = object_pose[2]

                    self.filter.update_object_config_history(object['id'], object)
                    if len(list(self.filter.object_config_history.get(object['id'], {'x': []})['x'])) >= 5:
                        object = self.filter.filter_pose(object['id'], object, 'object')

                    objects.append(object)

                if len(self.__markers_id) == 0:
                    for marker in markers.values():
                        self.__markers_id.add(marker['marker_id'])

                msg = 'Configuration successfully extracted.'
            else:
                msg = 'Wrong number of the markers or rigid bodies!'

        return agents, objects, filtered_markers if 'filtered_markers' in locals() else markers, msg
    
    
    def __unpackData(self) -> tuple[dict, dict]:
        if self.data is None:
            # raise ValueError('No data received from Motive!')
            return None, None
        
        rigid_body_data = self.data.rigid_body_data
        labeled_marker_data = self.data.labeled_marker_data

        labeled_marker_list = labeled_marker_data.labeled_marker_list
        rigid_body_list = rigid_body_data.rigid_body_list

        markers = {}
        rigid_bodies = {}

        for marker in labeled_marker_list:
            model_id, marker_id = [int(i) for i in marker.get_id()] 
            if len(self.__markers_id) != 0 and marker_id not in self.__markers_id:
                continue
            marker = {'model_id': model_id, 'marker_id': marker_id,
                        'marker_x': -marker.pos[0], 'marker_y': marker.pos[2], 'marker_z': marker.pos[1]}
            markers[str(model_id) + '.' + str(marker_id)] = marker

        for rigid_body in rigid_body_list:
            rigid_body = {'id': int(rigid_body.id_num), 'x': -rigid_body.pos[0], 'y': rigid_body.pos[2], 'z': rigid_body.pos[1], 'a': rigid_body.rot[0],
                            'b': rigid_body.rot[1], 'c': rigid_body.rot[2], 'd': rigid_body.rot[3]}
            rigid_bodies[int(rigid_body['id'])] = rigid_body    

        return markers, rigid_bodies
    
    def __simulateData(self) -> tuple[dict, dict]:
        pose = 1

        markers_df = pd.read_csv('Data/markers.csv')
        rigid_bodies_df = pd.read_csv('Data/rigid_bodies.csv')

        markers_df_ = markers_df[markers_df["pose"]
                                    == pose].drop('pose', axis=1)
        rigid_bodies_df_ = rigid_bodies_df[rigid_bodies_df["pose"] == pose].drop(
            'pose', axis=1)

        markers = {}
        rigid_bodies = {}

        for index, row in markers_df_.iterrows():
            marker = row.to_dict()
            marker['model_id'] = int(marker['model_id'])
            marker['marker_id'] = int(marker['marker_id'])
            markers[str(marker['model_id']) + '.' + str(marker['marker_id'])] = marker

        for index, row in rigid_bodies_df_.iterrows():
            rigid_body = row.to_dict()
            rigid_body['id'] = int(rigid_body['id'])
            rigid_bodies[rigid_body['id']] = rigid_body        

        return markers, rigid_bodies
    
    def __convertData(self, rigid_bodies: dict):
        for rigid_body in rigid_bodies.values():
            new_params = self.__quaternionToGlobal([rigid_body.get(param) for param in rb_params])
            for i in range(4):
                rigid_body[rb_params[i]] = new_params[i]

            # Convert quaternions to Euler angles
            euler_angles = self.__quaternionToEuler([rigid_body.get(param) for param in rb_params])
            for i in range(3):
                rigid_body[rb_angles[i]] = euler_angles[i]

    def __quaternionToGlobal(self, args):
        return [-args[0], args[2], args[1], args[3]]

    def __quaternionToEuler(self, coeffs):
        t0 = +2.0 * (coeffs[3] * coeffs[0] + coeffs[1] * coeffs[2])
        t1 = +1.0 - 2.0 * (coeffs[0] * coeffs[0] + coeffs[1] * coeffs[1])
        roll_x = np.arctan2(t0, t1)

        t2 = +2.0 * (coeffs[3] * coeffs[1] - coeffs[2] * coeffs[0])
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = np.arcsin(t2)

        t3 = +2.0 * (coeffs[3] * coeffs[2] + coeffs[0] * coeffs[1])
        t4 = +1.0 - 2.0 * (coeffs[1] * coeffs[1] + coeffs[2] * coeffs[2])
        yaw_z = np.arctan2(t3, t4)

        return [roll_x, pitch_y, yaw_z]  # in radians
    
    def __calcRbPose(self, rb, markers, tp='head') -> List[float]:
        points = []

        for marker in markers:
            points.append([marker['marker_x'], marker['marker_y']])

        points = np.array(points)
        triangle_sides = np.linalg.norm(points - np.roll(points, -1, 0), axis=1)

        i = triangle_sides.argsort()[1]
        i_next = i + 1 if i < 2 else 0

        if triangle_sides[i-1] > triangle_sides[i_next]:
            i, i_next = i_next, i

        delta = points[i_next, :] - points[i, :]
        theta = np.arctan2(delta[1], delta[0])
        # theta = theta  % (2 * np.pi)
        
        if tp == 'head':
            x = rb['x'] + global_var.HEAD_CENTER_R * np.cos(theta + global_var.HEAD_CENTER_ANGLE)
            y = rb['y'] + global_var.HEAD_CENTER_R * np.sin(theta + global_var.HEAD_CENTER_ANGLE)
        elif tp == 'manip':
            x = rb['x']
            y = rb['y']

        return [x, y, theta]
    
    def __rankPoints(self, markers, head_position) -> List[Marker]:
        head_position = np.array(head_position)

        # Define remaining points and their id's that correspond to thei original indicies
        remaining_points = [marker for marker in markers.values() if marker['model_id'] == 0]
        for remaining_point in remaining_points:
            remaining_point['pos'] = np.array([remaining_point.get(coord) for coord in pos_2d])

        # List of ranked points
        ranked_markers = []

        # Find the point closest to the head LU
        distances = np.linalg.norm([point['pos'] for point in remaining_points] - head_position, axis=1)

        current_index = distances.argmin()

        # Find the rank of the rest of the points
        while len(ranked_markers) < 6 and remaining_points:
            current_point = remaining_points.pop(current_index)
            vsf_marker = Marker(current_point['marker_id'], current_point['marker_x'], current_point['marker_y'])
            ranked_markers.append(vsf_marker)

            if(remaining_points):
                distances = np.linalg.norm(
                    [point['pos'] for point in remaining_points] - current_point['pos'], axis=1)
                current_index = distances.argmin()

        return ranked_markers
    
    def __getAngle(self, p1: list, p2: list):
        a = p2[0] - p1[0]
        b = p2[1] - p1[1]

        alpha = np.arctan2(b, a)

        return alpha
    
    def __extrapolateCurve(self, points:List[Marker]) -> tuple[float, float, float, float, float]:
        x = []
        y = []

        for point in points:
            x.append(point.x)
            y.append(point.y)
            
        # Fit a spline curve to the data points
        tck, _ = splprep([x, y], s=0)

        # Calculate the arc length of the spline curve
        s = np.linspace(0, 1, 1000)
        spline_x, spline_y = splev(s, tck)
        spline_arc_length = 0
        for i in range(1, len(s)):
            spline_arc_length += np.sqrt((spline_x[i] - spline_x[i-1])**2 + (spline_y[i] - spline_y[i-1])**2)

        # Find the midpoint of the spline curve
        spline_arc_length_midpoint = spline_arc_length / 2
        current_arc_length = 0
        for i in range(1, len(s)):
            current_arc_length += np.sqrt((spline_x[i] - spline_x[i-1])**2 + (spline_y[i] - spline_y[i-1])**2)
            if current_arc_length >= spline_arc_length_midpoint:
                midpoint_s = s[i-1]
                midpoint_x, midpoint_y = splev(midpoint_s, tck)
                break

        # Calculate the tangent vector at the midpoint
        tangent_x, tangent_y = splev(midpoint_s, tck, der=1)
        theta = np.arctan2(tangent_y, tangent_x)

        if self.agent_theta_prev is None:
            self.agent_theta_prev = theta
        else:
            diff = theta - self.agent_theta_prev
            # Check if we've crossed the -pi/pi boundary
            offset = 0
            if diff > np.pi:
                offset -= 2*np.pi
            elif diff < -np.pi:
                offset += 2*np.pi

            theta += offset
            self.agent_theta_prev = theta

        x_der1, y_der1 = splev(s[:500], tck, der=1)
        x_der2, y_der2 = splev(s[:500], tck, der=2)

        curvature = (x_der1 * y_der2 - y_der1 * x_der2) / np.power(x_der1** 2 + y_der1** 2, 3 / 2)
        k1 = curvature.mean()
        
        x_der1, y_der1 = splev(s[500:], tck, der=1)
        x_der2, y_der2 = splev(s[500:], tck, der=2)

        curvature = (x_der1 * y_der2 - y_der1 * x_der2) / np.power(x_der1** 2 + y_der1** 2, 3 / 2)
        k2 = curvature.mean()
    
        # If the curve has only one point, return that point
        return midpoint_x.item(), midpoint_y.item(), theta, k1, k2