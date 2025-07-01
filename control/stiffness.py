"""
    This controller manages the state transitions for the robot's variable stiffness 
    bridge by implementing a Finite-State Machine (FSM) for each segment to handle 
    the switch between Rigid (0) and Flexible (1) states. The electrical circuit is 
    designed to approximate the stiffness of an entire segment using a single 
    temperature sensor, which causes a hysteresis effect. To compensate for it, two 
    different temperature thresholds are employed.
"""

from entities import robot_state

class FSM:
    def __init__(self, robot: robot_state.Model):
        """
        Initializes the FSM controller.

        Args:
            robot (robot_state.Model): An instance of the robot's state model.
        """

        self.robot = robot
        # The "upper" threshold (62°C) for confirming a transition to the Flexible state.
        self.liquid_threshold = 62
        # The "lower" threshold (53°C) for confirming a transition to the Rigid state.
        self.solid_threshold = 53

        # --- Declarative FSM Rules Definition ---
        self.FSM_RULES = {
            # Rule for: Current State -> Target State
            # (current_state, target_state): {rule definition}

            # --- Transition to Flexible ---
            (0, 1): {
                "action_name": "TO FLEXIBLE",
                "action_value": 1,
                "condition": lambda r, i: r.temp[i] >= self.liquid_threshold,
                "effect": lambda r, i: self._set_stiffness(i, 1),
                "message": "Heating segment {idx} to become flexible... (Temp: {temp}°C)"
            },
            # --- Transition to Rigid ---
            (1, 0): {
                "action_name": "TO RIGID",
                "action_value": -1,
                "condition": lambda r, i: r.temp[i] <= self.solid_threshold,
                "effect": lambda r, i: self._set_stiffness(i, 0),
                "message": "Cooling segment {idx} to become rigid... (Temp: {temp}°C)"
            },
            # --- Maintain Current State (No change needed) ---
            (0, 0): {"action_name": "NONE", "action_value": 0},
            (1, 1): {"action_name": "NONE", "action_value": 0},
        }

    def main(self, target_states: list) -> tuple:
        """
        Determines and applies necessary actions based on the FSM rules.

        This method iterates through each robot segment, determines the required 
        action from the FSM rules, and processes the state transition logic 
        (checking conditions and applying effects).

        Args:
            target_states (list): The desired stiffness states, e.g., [1, 0].

        Returns:
            tuple[bool, tuple]: A tuple containing:
                - is_transitioning (bool): True if any action is being taken.
                - actions (tuple): The numeric actions being applied, e.g., (1, -1).
        """

        current_states = self.robot.stiffness
        actions_list = []
        
        for i, target_state in enumerate(target_states):
            rule =  self.FSM_RULES.get((current_states[i], target_state))

            action_value = rule.get("action_value", 0)
            actions_list.append(action_value)

            if action_value == 0:
                continue

            # --- Process the state transition ---
            # Check the condition defined in the rule (e.g., has temp passed the threshold?).
            if rule["condition"](self.robot, i):
                # If the condition is met, execute the effect (e.g., change the stiffness).
                rule["effect"](self.robot, i)
                print(f"Success: Segment {i+1} is now {rule['action_name'].lower()}.")
            else:
                # If not met, print the progress message.
                print(rule["message"].format(idx=i+1, temp=self.robot.temp[i]))
            print()

        actions = tuple(actions_list)
        is_transitioning = any(action != 0 for action in actions)

        return is_transitioning, actions

    def _set_stiffness(self, segment_index: int, value: int):
        """Dynamically set stiffness on the robot state object"""
        if segment_index == 0:
            self.robot.stiff1 = value
        else:
            self.robot.stiff2 = value
