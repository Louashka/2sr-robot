from entities import global_var as gv, robot_state

class GraspPlan():
    def __init__(self, robot: robot_state.Model):
        self.robot = robot

    def get_grasp_phases(self) -> tuple:
        approach = ...
        pre_grasp = ...
        final_contact = ...
        
        return approach, pre_grasp, final_contact