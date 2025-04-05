import env, Controller.transport as trans, Controller.traverse as trav
import numpy as np
import json
import time

title = 'grasp_trial_1'

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
                'optimal_pre_grasp': None
            },
            'approach': None,
            'pre_grasp': None,
            'final_contact': None,
            'transport': None
        }

    def addDate(self, date_input):
        self.data_json['metadata']['date'] = date_input

    def addApproachData(self, approach_data):
        self.data_json['approach'] = approach_data

    def addPreGraspData(self, grasp_data):
        self.data_json['pre_grasp'] = grasp_data

    def addFinalContactData(self, f_contact_data):
        self.data_json['final_contact'] = f_contact_data

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
    grasp_ctr_point, approach_target, pre_grasp, final_contact = trans.defineGrasp(env_observer.object)

    # env_observer.rgb_camera.grasp_point = pre_grasp[:2]
    env_observer.rgb_camera.heading = env_observer.object.heading_angle
    print(pre_grasp)

    # --------------------------------- Approach --------------------------------
    print('Approach the object...\n')
    elapsed_time = 0
    start_time = time.perf_counter()

    traverse_data, end_time, vel = trav.traverseObstacles(env_observer.agent, env_observer.object,
                [approach_target], start_time, env_observer.rgb_camera, simulation)
    
    data_collector.addApproachData(traverse_data)

    # import matplotlib.pyplot as plt

    # plt.figure(figsize=(15, 4))

    # plt.subplot(1, 3, 1)
    # plt.plot(vel[0], label='v_x')
    # plt.legend()

    # plt.subplot(1, 3, 2)
    # plt.plot(vel[1], label='v_y')
    # plt.legend()

    # plt.subplot(1, 3, 3)
    # plt.plot(vel[2], label='omega')
    # plt.legend()

    # plt.tight_layout()
    # plt.show()

    # dj_raw, dj_filtered = env_observer.mocap.filter.calculate_dimensionless_jerk(1)
    # if dj_raw is not None:
    #     improvement = 100 * (dj_raw - dj_filtered) / dj_raw
    #     print(f"Dimensionless jerk reduced by {improvement:.1f}%")
    #     print(f"Raw: {dj_raw:.2f}, Filtered: {dj_filtered:.2f}")

    # env_observer.mocap.filter.create_window_evaluation_plot(1)

    # -------------------------------- Pre-grasp --------------------------------
    print('\nPre-grasp the object...\n')
    pre_grasp_data, end_time, vel = trav.traverseObstacles(env_observer.agent, env_observer.object,
                [pre_grasp], start_time, env_observer.rgb_camera, simulation)
    
    data_collector.addPreGraspData(pre_grasp_data)

    # ------------------------------- Final contact -------------------------------
    print('\Contact the object...\n')
    f_contact_data, end_time, vel = trav.traverseObstacles(env_observer.agent, env_observer.object,
                [final_contact], start_time, env_observer.rgb_camera, simulation)
    
    data_collector.addFinalContactData(f_contact_data)

    # -------------------------------- Transport --------------------------------
    trp_target = env_observer.agent.config.tolist()
    trp_target[0] -= 0.5 * np.sin(trp_target[2] - np.pi/6)
    trp_target[1] += 0.5 * np.cos(trp_target[2])
    trp_target[2] -= np.pi/6
    
    print('\nTransport the object...\n')
    transport_data, end_time, vel = trav.traverseObstacles(env_observer.agent, env_observer.object,
                [trp_target], start_time, env_observer.rgb_camera, simulation)
    
    data_collector.addTransportData(transport_data)

    # ---------------------------------------s------------------------------------
    print('Save data...')
    data_file_path = f'Experiments/Data/Tracking/Grasp/{env_observer.date_title}.json'
    data_collector.saveData(data_file_path, False)

