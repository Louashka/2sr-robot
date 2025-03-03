import env, Controller.transport as trans, Controller.traverse as trav
import numpy as np
import json
import time

title = 'grasp_exp_heart'

# simulation = True
simulation = False

class DataCollector:
    def __init__(self): 
        self.data_json = {
            'metadata': {
                'description': 'Full-body grasp experiment',
                'date': None
            },
            'estimation': {
                'optimal_contact_point': None,
                'optimal_grasp_config': None
            },
            'approach': None,
            'grasp': None,
            'transport': None
        }

    def addDate(self, date_input):
        self.data_json['metadata']['date'] = date_input

    def addApproachData(self, approach_data):
        self.data_json['approach'] = approach_data

    def addGraspData(self, grasp_data):
        self.data_json['grasp'] = grasp_data

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
    env_observer = env.Observer(title, simulation)
    env_observer.run()

    data_collector = DataCollector()
    data_collector.addDate(env_observer.date_title)

    # ---------------------- Determine Grasping Parameters ----------------------
    env_observer.object.delta_theta = np.pi/2 - env_observer.object.theta
    grasp_ctr_point, grasp_config, approach_target = trans.defineGrasp(env_observer.object)

    # env_observer.rgb_camera.grasp_point = grasp_config[:2]
    env_observer.rgb_camera.heading = env_observer.object.heading_angle
    print(approach_target)

    # --------------------------------- Approach --------------------------------
    print('Approach the object...\n')
    elapsed_time = 0
    start_time = time.perf_counter()

    traverse_data, end_time = trav.traverseObstacles(env_observer.agent, env_observer.object,
                [approach_target], start_time, env_observer.rgb_camera, simulation)
    
    data_collector.addApproachData(traverse_data)

    # ---------------------------------- Grasp ----------------------------------
    print('\nGrasp the object...\n')
    grasp_data, end_time = trav.traverseObstacles(env_observer.agent, env_observer.object,
                [grasp_config], start_time, env_observer.rgb_camera, simulation)
    
    data_collector.addGraspData(grasp_data)

    # -------------------------------- Transport --------------------------------
    trp_target = env_observer.agent.config.tolist()
    trp_target[0] -= 0.5 * np.sin(trp_target[2] - np.pi/6)
    trp_target[1] += 0.5 * np.cos(trp_target[2])
    trp_target[2] -= np.pi/6
    
    print('\nTransport the object...\n')
    transport_data, end_time = trav.traverseObstacles(env_observer.agent, env_observer.object,
                [trp_target], start_time, env_observer.rgb_camera, simulation)
    
    data_collector.addTransportData(transport_data)

    # ---------------------------------------s------------------------------------
    print('Save data...')
    data_file_path = f'Experiments/Data/Tracking/Grasp/{env_observer.date_title}.json'
    data_collector.saveData(data_file_path, False)

