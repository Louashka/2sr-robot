import env, Controller.transport as trans, Controller.traverse as trav
import numpy as np
import json
import time
from Model import global_var as gv
from View import plotlib

obj_dir = np.pi/2
obj_go_dist = 0.7

# simulation = True
simulation = False

class DataCollector:
    def __init__(self): 
        self.data_json = {
            'metadata': {
                'description': 'Traverse a cluttered environment (adaptive morphology)',
                'date': None
            },
            'estimation': {
                'grasp_config': None,
                'points_paths': None,
                'points_paths_adjusted': None,
                'orientations': None,
                'robot_reference_path': None,
                'key_configs': None
            },
            'traversing': None,
            'transport': None
        }

    def addDate(self, date_input):
        self.data_json['metadata']['date'] = date_input

    def addGraspConfig(self, grasp_config):
        self.data_json['estimation']['grasp_config'] = grasp_config

    def addPaths(self, paths):
        front_path, middle_path, rear_path = paths
        self.data_json['estimation']['points_paths'] = {
            'front': front_path,
            'middle': middle_path,
            'rear': rear_path
        }

    def addAdjustedPaths(self, adjusted_paths):
        front_adjusted_path, middle_adjusted_path, rear_adjusted_path = adjusted_paths
        self.data_json['estimation']['points_paths_adjusted'] = {
            'front': front_adjusted_path,
            'middle': middle_adjusted_path,
            'rear': rear_adjusted_path
        }

    def addOrientations(self, orientations):
        self.data_json['estimation']['orientations'] = orientations

    def addRefPath(self, ref_path):
        ref = [ref_point.tolist() for ref_point in ref_path]
        self.data_json['estimation']['robot_reference_path'] = ref

    def addCrucialConfigs(self, key_configs):
        self.data_json['estimation']['key_configs'] = key_configs

    def addTraversingData(self, traverse_data):
        self.data_json['traversing'] = traverse_data

    def addTransportData(self, transport_data):
        self.data_json['transport'] = transport_data

    def saveData(self, file_path, simulation=False):
        """Save collected data to JSON file"""
        if not simulation:
            with open(file_path, 'w') as f:
                json.dump(self.data_json, f, indent=2)
            print(f"Data written to {file_path}\n")


if __name__ == "__main__":
    # Set up the environment 
    env_observer = env.Observer(simulation)
    env_observer.run()

    # Detect obstacles
    env_observer.detectObstacles()

    # Define data collector
    data_collector = DataCollector()
    data_collector.addDate(env_observer.date_title)

    # ---------------------- Generate object's target path ----------------------
    
    # Target position of the object
    obj_target_pos = [env_observer.object.x + obj_go_dist * np.cos(obj_dir),
                      env_observer.object.y + obj_go_dist * np.sin(obj_dir)]

    obj_path, obj_path_points, obj_d_th, obj_target_th = trans.generatePath(env_observer.object.pose, 
                                                obj_target_pos, obj_dir, np.pi/2-obj_dir)
    env_observer.object.delta_theta = obj_d_th
    env_observer.defineTargetObject([*obj_target_pos, obj_target_th])
    env_observer.showObjPath(obj_path_points)

    # Grasp parameters
    grasp_idx, grasp_config, traverse_target = trans.defineGrasp(env_observer.object)
    data_collector.addGraspConfig(grasp_config)
    # env_observer.rgb_camera.target_robot_config = grasp_config

    # ------------------------- Conduct Motion Planning -------------------------
    
    # Build a Voronoi diagram and find paths of the 3 key robot's points 
    front_path, middle_path, rear_path, (fig, ax) = trav.voronoiAnalysis(env_observer.expanded_obstacles,
                                                                env_observer.agent.pose, traverse_target[:3])
    data_collector.addPaths([front_path, middle_path, rear_path])

    # Adjust each path
    front_seg_idx, front_path_adjusted = trav.adjustPath(front_path)
    middle_seg_idx, middle_path_adjusted = trav.adjustPath(middle_path)
    rear_seg_idx, rear_path_adjusted = trav.adjustPath(rear_path)
    
    data_collector.addAdjustedPaths([front_path_adjusted, middle_path_adjusted, rear_path_adjusted])

    plotlib.plotDotPaths((front_path, middle_path, rear_path),
                         (front_path_adjusted, middle_path_adjusted, rear_path_adjusted),
                         (front_seg_idx, middle_seg_idx, rear_seg_idx))

    theta_seq = trav.calcOrientations(middle_path_adjusted, rear_path_adjusted, middle_seg_idx[1:-1])
    data_collector.addOrientations(theta_seq)

    # Fit robot's configuration
    print('Calculate the robot\'s reference path...\n')

    fitter = trav.RobotConfigurationFitter(gv.L_VSS, gv.L_CONN + gv.LU_SIDE)
    q_ref = fitter.fit_configurations(middle_path_adjusted, front_path_adjusted,
                                      rear_path_adjusted, theta_seq)
    data_collector.addRefPath(q_ref)

    print('Determine crucial shapes...\n')
    key_configs_idx = trav.getCrucialConfigs(q_ref)
    data_collector.addCrucialConfigs(key_configs_idx)


    '''
    Animate the estimation results
    '''

    key_configs_anim = []

    for i in range(len(key_configs_idx)-1):
        key_configs_anim.extend([q_ref[key_configs_idx[i]]] * (key_configs_idx[i+1] - key_configs_idx[i]))
    # key_configs_anim.append(q_ref[key_configs_idx[-1]])
    key_configs_anim.extend([q_ref[key_configs_idx[-1]]]*(len(rear_path) - len(key_configs_anim)))
    
    
    # print('Start animation...\n')

    # trav.runAnimation(fig, rear_path_adjusted, front_path_adjusted, middle_path_adjusted, theta_seq, 
    #                   q_ref, key_configs_anim)
    
    # --------------------------- Execute Experiment ----------------------------

    continue_flag = False
    
    user_input = input("Proceed? (y/n): ").strip().lower()
    if user_input == 'y' or user_input == 'yes':
        print("Proceeding with the experiment!\n")
        continue_flag = True
    elif user_input == 'n' or user_input == 'no':
        print("Operation cancelled.\n")
    else:
        print("Invalid input. Please enter 'y' or 'n'.\n")


    if continue_flag:
        '''
        Discretize robot target configuration from identified key configurations
        '''
        
        key_configs = [q_ref[i] for i in key_configs_idx]

        # start_config = key_configs[0]
        # zero_config = np.array([start_config[0]-0.0, start_config[1]-0.3, 
        #                         start_config[2]-1.5, 0, 0])
        # key_configs = [zero_config] + key_configs


        print('Key configurations:')
        print(key_configs)

        all_target_configs = trav.discretizeConfigs(env_observer.agent.config, key_configs)
        all_target_configs.append(grasp_config)
        print('\nDiscrete target configurations:')
        print(all_target_configs)

        print('\nTraverse obstacles...\n')
        elapsed_time = 0
        start_time = time.perf_counter()
        traverse_data, end_time = trav.traverseObstacles(env_observer.agent, 
                    all_target_configs, start_time, env_observer.rgb_camera, simulation)
        data_collector.addTraversingData(traverse_data)

        print('\nTransport the object...\n')
        transport_data = trans.transport(env_observer.agent, env_observer.object, 
        env_observer.object_target.position, obj_path, env_observer.rgb_camera, start_time, simulation)
        data_collector.addTransportData(transport_data)
    
        print('Experiment is finished!\n')

    print('Save data...')
    data_file_path = f'Experiments/Data/Tracking/Grasp/traverse_sm_{env_observer.date_title}.json'
    data_collector.saveData(data_file_path, False)



