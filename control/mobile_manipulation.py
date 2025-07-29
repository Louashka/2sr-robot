import numpy as np
from entities import manipulandum, global_var as gv

class ObjectHandlingPolicy:
    def __init__(self, delta_fc=0.0, delta_pg=0.02, delta_app=0.04) -> None:
        self.delta_fc = delta_fc # Distance between the object and the final contact
        self.delta_pg = delta_pg # Distance between the object and the pre-grasp configuration
        self.delta_app = delta_app # Distance between the object and the approach configuration

    def grasp(self, object: manipulandum.Shape) -> tuple:
        dir_angle = object.heading - np.pi
        direction_vector = np.array([np.cos(dir_angle), np.sin(dir_angle)])

        # Find a point on the contour in the opposite direction
        s_array = np.linspace(0, 1, 200)
        max_dot_product = 0
        grasp_idx = 0

        for s in s_array:
            point = object.get_point(s)
            theta = object.getTangent(s)
            
            vector_to_point = np.array(point) - object.position
            dot_product = np.dot(vector_to_point, direction_vector)
            
            if dot_product > max_dot_product:
                grasp_idx = s
                optimal_theta = theta
                max_dot_product = dot_product

                final_contact_pos = [point[0] + self.delta_fc * np.cos(dir_angle), 
                                    point[1] + self.delta_fc * np.sin(dir_angle)]

                pre_grasp_pos = [point[0] + self.delta_pg * np.cos(dir_angle), 
                                point[1] + self.delta_pg * np.sin(dir_angle)]
                
                approach_pos = [point[0] + self.delta_app * np.cos(dir_angle), 
                                point[1] + self.delta_app * np.sin(dir_angle)]

        k1 = object.getMeanCurvature(grasp_idx, gv.L_VSS, 'clockwise')
        k2 = object.getMeanCurvature(grasp_idx, gv.L_VSS)

        approach = [*approach_pos, optimal_theta, 0, 0]
        pre_grasp = [*pre_grasp_pos, optimal_theta, k1, k2]
        final_contact = [*final_contact_pos, optimal_theta, k1, k2]
        
        return grasp_idx, approach, pre_grasp, final_contact
    
    def transport(self):
        return