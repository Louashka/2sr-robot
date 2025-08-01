import numpy as np
from entities import manipulandum, global_var as gv

class ObjectHandlingPolicy:
    def __init__(self, delta_fc=0.0, delta_pg=0.02, delta_app=0.04) -> None:
        self.delta_fc = delta_fc # Distance between the object and the final contact
        self.delta_pg = delta_pg # Distance between the object and the pre-grasp configuration
        self.delta_app = delta_app # Distance between the object and the approach configuration

        self.L_ROBOT = gv.L_VSB + 2 * gv.L_CONN + gv.LU_SIDE

    def grasp(self, object: manipulandum.Shape) -> tuple:
        dir_angle = object.heading - np.pi
        direction_vector = np.array([np.cos(dir_angle), np.sin(dir_angle)])

        # Find a point on the contour in the opposite direction
        s_array = np.linspace(0, 1, 200)
        max_dot_product = 0
        grasp_idx = 0

        for s in s_array:
            point = object.get_point(s)
            theta = object.get_tangent(s)
            
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

        k1 = object.get_mean_curvature(grasp_idx, gv.L_VSS, 'clockwise')
        k2 = object.get_mean_curvature(grasp_idx, gv.L_VSS)

        approach = [*approach_pos, optimal_theta, 0, 0]
        pre_grasp = [*pre_grasp_pos, optimal_theta, k1, k2]
        final_contact = [*final_contact_pos, optimal_theta, k1, k2]
        
        return grasp_idx, approach, pre_grasp, final_contact
    
    def calculate_force_closure_potential(self, shape: manipulandum.Shape, s1: float, s2: float,
                                          w_force: float = 0.0, w_length: float = 1.0) -> float:
        """
        Calculates the force closure attraction potential for two points on the contour.
        
        Args:
            shape: The object instance.
            s1: The parameter [0, 1] for the first contact point (P1).
            s2: The parameter [0, 1] for the second contact point (P2).
            
        Returns:
            The potential value, from 0 (best) to 2 (worst).
        """
        # --- 1. Force Closure Potential (U_force) ---
        # 1. Get the Center of Mass (Centroid)
        com = shape.centroid
        
        # 2. Get the two contact points on the contour
        p1 = shape.get_point(s1)
        p2 = shape.get_point(s2)
        
        # 3. Calculate the vectors v1 (P1->COM) and v2 (COM->P2)
        v1 = com - np.array(p1)
        v2 = np.array(p2) - com
        
        # 4. Normalize the vectors. Handle the edge case where a point is at the COM.
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if np.isclose(norm_v1, 0) or np.isclose(norm_v2, 0):
            return float('inf')
            
        v1_norm = v1 / norm_v1
        v2_norm = v2 / norm_v2
        
        # 5. Calculate the dot product and the final potential
        dot_product = np.dot(v1_norm, v2_norm)
        potential_force = 1.0 - dot_product

        # --- 2. Length Constraint Potential (U_length) ---
        # MODIFIED: Calculate the counter-clockwise path distance from s1 to s2.
        # path_distance_norm = (s2 - s1 + 1.0) % 1.0
        ds = abs(s1 - s2)
        path_distance_norm = min(ds, 1.0 - ds)
        
        actual_contour_length = path_distance_norm * shape.total_arc_length
        length_error = actual_contour_length - self.L_ROBOT
        potential_length = (length_error / self.L_ROBOT)**2

        # --- 3. Combine Potentials ---
        return w_force * potential_force + w_length * potential_length