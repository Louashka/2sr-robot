import env, Controller.transport as trans, Controller.traverse as trav
import numpy as np
import json
import time
from Model import global_var as gv

obj_dir = np.pi/2
obj_go_dist = 0.7

# simulation = True
simulation = False

class DataCollector:
    def __init__(self): 
        self.data_json = {
            'metadata': {
                'description': 'Traverse a cluttered environment (RRT* + APF)',
                'date': None
            },
            'estimation': {
                'grasp_config': None,
                'robot_reference_path': None,
            },
            'traversing': None,
            'transport': None
        }

    def addDate(self, date_input):
        self.data_json['metadata']['date'] = date_input

    def addGraspConfig(self, grasp_config):
        self.data_json['estimation']['grasp_config'] = grasp_config

    def addRefPath(self, ref_path):
        self.data_json['estimation']['robot_reference_path'] = ref_path

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
    env_observer.rgb_camera.target_robot_config = grasp_config

    # ------------------------- Conduct Motion Planning -------------------------

    print('Start path planning...\n')
    path = trav.runRigidPlanner(env_observer.agent.pose, traverse_target[:3], 
        grasp_config[:3], env_observer.expanded_obstacles, env_observer.rgb_camera)
    
    if path is not None: 
        # Defind target configurations
        all_target_configs = []
        for p in path:
            all_target_configs.append(p + [0, 0])
        all_target_configs.append(grasp_config)
        data_collector.addRefPath(all_target_configs)

        env_observer.rgb_camera.rrt_path = None
        print('\nTraverse obstacles...\n')
        elapsed_time = 0
        start_time = time.perf_counter()
        traverse_data, end_time = trav.traverseObstacles(env_observer.agent, 
                    all_target_configs[1:], start_time, env_observer.rgb_camera, simulation)
        data_collector.addTraversingData(traverse_data)

        print('\nTransport the object...\n')
        transport_data = trans.transport(env_observer.agent, env_observer.object, 
        env_observer.object_target.position, obj_path, env_observer.rgb_camera, start_time, simulation)
        data_collector.addTransportData(transport_data)
    
        print('Experiment is finished!\n')

    print('Save data...')
    data_file_path = f'Experiments/Data/Tracking/Grasp/traverse_rm_{env_observer.date_title}.json'
    data_collector.saveData(data_file_path, False)

