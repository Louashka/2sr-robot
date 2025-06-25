import env, transport as trans, traverse as trav
import robot2sr_controller as rsr_ctrl
import numpy as np
import json
import time
import serial

title = 'grasp_bean_2'

# simulation = True
simulation = False

port_name = "COM3"
serial_port = serial.Serial(port_name, 115200)

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
    env_observer = env.Observer(title, serial_port, simulation)
    env_observer.run()

    data_collector = DataCollector()
    data_collector.addDate(env_observer.date_title)

    agent_controller = rsr_ctrl.Controller(serial_port)

    # ---------------------------------- Determine Grasping Parameters ----------------------------------
    env_observer.object.delta_theta = np.pi/2.8 - env_observer.object.theta
    grasp_ctr_point, approach_target, pre_grasp, final_contact = trans.defineGrasp(env_observer.object)

    # env_observer.rgb_camera.grasp_point = pre_grasp[:2]
    env_observer.rgb_camera.heading = env_observer.object.heading_angle
    print(pre_grasp)

    # --------------------------------------------- Approach --------------------------------------------
    print('Approach the object...\n')
    elapsed_time = 0
    start_time = time.perf_counter()

    traverse_data, end_time, vel = trav.traverseObstacles(env_observer.agent, env_observer.object,
                agent_controller, [approach_target], start_time, env_observer.rgb_camera, False, simulation)
    
    data_collector.addApproachData(traverse_data)

    # -------------------------------------------- Pre-grasp --------------------------------------------
    print('\nPre-grasp the object...\n')
    pre_grasp_data, end_time, vel = trav.traverseObstacles(env_observer.agent, env_observer.object,
                agent_controller, [pre_grasp], start_time, env_observer.rgb_camera, True, simulation)
    
    data_collector.addPreGraspData(pre_grasp_data)

    # ------------------------------------------ Final contact ------------------------------------------
    print('\nContact the object...\n')
    f_contact_data, end_time, vel = trav.traverseObstacles(env_observer.agent, env_observer.object,
                agent_controller, [final_contact], start_time, env_observer.rgb_camera, False, simulation)
    
    data_collector.addFinalContactData(f_contact_data)

    # -------------------------------------------- Transport --------------------------------------------
    obj_dir = np.pi/2.8
    obj_go_dist = 0.8

    obj_target_pos = [env_observer.object.x + obj_go_dist * np.cos(obj_dir),
                      env_observer.object.y + obj_go_dist * np.sin(obj_dir)]

    obj_path, obj_path_points, obj_d_th, obj_target_th = trans.generatePath(env_observer.object.pose, 
                                                obj_target_pos, np.pi/4, np.pi/7)
    env_observer.object.delta_theta = obj_d_th
    env_observer.defineTargetObject([*obj_target_pos, obj_target_th])
    env_observer.showObjPath(obj_path_points)

    
    print('\nTransport the object...\n')
    transport_data = trans.transport(env_observer.agent, env_observer.object, agent_controller,
        env_observer.object_target.position, obj_path, env_observer.rgb_camera, start_time, simulation)
    
    data_collector.addTransportData(transport_data)

    # ---------------------------------------------- Save -----------------------------------------------
    print('Save data...')
    data_file_path = f'Experiments/Data/Tracking/Grasping/{env_observer.date_title}.json'
    data_collector.saveData(data_file_path, False)

