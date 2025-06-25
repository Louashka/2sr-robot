"""
    This controller manages the state transitions for the robot's variable stiffness 
    bridge by implementing a Finite-State Machine (FSM) for each segment to handle 
    the switch between Rigid (0) and Flexible (1) states. The electrical circuit is 
    designed to approximate the stiffness of an entire segment using a single 
    temperature sensor, which causes a hysteresis effect. To compensate for it, two 
    different temperature thresholds are employed.
"""

from entities import robot

class FSMController:
    def __init__(self, robot: robot.Model):
        self.robot = robot

        # The "upper" threshold (62°C) for confirming a transition to the Flexible state.
        self.liquid_threshold = 62
        # The "lower" threshold (53°C) for confirming a transition to the Rigid state.
        self.solid_threshold = 53

    def main(self, target_states: list) -> bool:
        """
        Determines the necessary actions, applies them, and reports whether a state 
        transition is in progress.

        Args:
            target_states (list): The desired stiffness states, e.g., [1, 0].

        Returns:
            tuple[bool, tuple]: A tuple containing:
                - status (bool): True if a transition is in progress, False otherwise.
                - actions (tuple): The actions being applied, e.g., (1, -1).
        """

        actions = self.__get_actions(target_states)
        self.__apply_actions(actions)

        status = True

        if actions == (0, 0):
            status = False 

        return status, actions

    def __get_actions(self, target_states):
        """
        Determines the actions for both segments based on their current and target states.
        """

        actions = (self.__get_action(self.robot.stiff1, target_states[0]),
                   self.__get_action(self.robot.stiff2, target_states[1]))
        
        return actions

    def __get_action(self, state, target_state):
        """
        Defines the FSM logic for a single segment. It returns the action needed to move 
        from the current state to the target state.
        
        Returns:
             1: Heat the segment (transition to Flexible)
            -1: Cool the segment (transition to Rigid)
             0: Maintain current state
        """

        if state == target_state:
            return 0
        elif state == 0 and target_state == 1:
            return 1
        else:
            return -1
        
    def __apply_actions(self, actions):
        """
        Applies the determined action to each segment by calling the helper method.
        """
        
        for i in range(len(actions)):
            self.__apply_action(i, actions[i])

    def __apply_action(self, i, action):
        """
        The core of the FSM's state transition logic with hysteresis.
        It checks if the temperature has crossed the correct threshold
        before officially changing the robot's state attribute.
        """

        if action == 1:
            if self.robot.temp[i] >= self.liquid_threshold:
                if i == 0: 
                    self.robot.stiff1 = 1
                else:
                    self.robot.stiff2 = 1
            else:
                print(f'Switching segment {i+1} to soft...')
                print(f'Current temp: {self.robot.temp[i]}')
                print()
        if action == -1:
            if self.robot.temp[i] <= self.solid_threshold:
                if i == 0: 
                    self.robot.stiff1 = 0
                else:
                    self.robot.stiff2 = 0
            else:
                print(f'Switching segment {i+1} to rigid...')
                print(f'Current temp: {self.robot.temp[i]}')
                print()