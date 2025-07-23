import numpy as np
import collections
import copy
from gekko import GEKKO
from entities import global_var as gv, robot_state
from control import stiffness
import kinematics

class Morphing:
    """
    Manages the complex, sequential morphing of a 2-segment 2SR robot.
    
    This controller handles the constraint that the robot cannot bend both
    segments in opposite directions simultaneously. It does this by creating
    and executing a sequential plan.
    """
    # Mapping from stiffness state to a descriptive control mode.
    STIFFNESS_TO_MODES = {
        (0, 0): 'RIGID_MOTION',
        (1, 0): 'SHAPE_MORPH_1',
        (0, 1): 'SHAPE_MORPH_2',
        (1, 1): 'SHAPE_MORPH_3',
    }

    def __init__(self, robot: robot_state.Model):
        self.robot = robot
        # The low-level FSM for handling stiffness transitions.
        self.stiffness_controller = stiffness.FSM(robot)
        
        # A deque is a list optimized for adding and removing items from the ends.
        # This will hold our sequence of (stiff1, stiff2) commands.
        self.morph_plan = collections.deque()
        
        self.k_threshold = 0.1  # Threshold to decide if a stiffness change is needed.

    def update_control_mode(self, target_k_config: list) -> tuple:
        """
        Determines and executes the next step in a morphing plan.
        It manages the plan and returns the current action.

        Args:
            target_k_config (list): The final target [k1, k2] values.

        Returns:
            A tuple: (mode, current_target_k, stiffness_transitions)
        """
        # --- Step 1: Check if the current step is complete ---
        if self.morph_plan and self._is_current_step_achieved():
            self.morph_plan.popleft()

        # --- Step 2: Create a plan if one doesn't exist ---
        if not self.morph_plan:
            self._create_morph_plan(target_k_config)

        # --- Step 3: Execute the current step of the plan ---
        current_k_target = self.morph_plan[0]['curvature']
        current_stiffness_target = self.morph_plan[0]['stiffness']

        # Use the low-level FSM to manage the transition to the required stiffness
        is_transitioning, stiff_transitions = self.stiffness_controller.main(current_stiffness_target)
        
        current_mode = self.STIFFNESS_TO_MODES.get(current_stiffness_target)

        if is_transitioning:
            current_mode = 'IDLE'
            
        return current_mode, current_k_target, stiff_transitions

    def _create_morph_plan(self, target_k_config: list):
        """
        Analyzes the target and creates a sequential plan of stiffness states.
        This is where the core logic for handling split-direction changes resides.
        """
        k1_current, k2_current = self.robot.k1, self.robot.k2
        k1_target, k2_target = target_k_config[0], target_k_config[1]

        # Determine if a change is needed for each segment
        needs_k1_change = abs(self.robot.k1 - k1_target) > self.k_threshold
        needs_k2_change = abs(self.robot.k2 - k2_target) > self.k_threshold
        
        # Determine the direction of change (1 for increase, -1 for decrease, 0 for no change)
        # Using a small epsilon to avoid issues with floating point comparisons
        k1_dir = int(np.sign(k1_target - self.robot.k1)) if needs_k1_change else 0
        k2_dir = int(np.sign(k2_target - self.robot.k2)) if needs_k2_change else 0
        
        # If both need to change
        if needs_k1_change and needs_k2_change:
            # Subcase 1.1: Split-direction change (e.g., k1 up, k2 down)
            if k1_dir != k2_dir:
                print("INFO: Split-direction change. Planning sequential morph.")
                # Plan A: Morph k1, then k2.
                self.morph_plan.append({
                    'curvature': (k1_target, k2_current),
                    'stiffness': (1, 0),
                    'priority': 0 # Segment with the highest priority in bending 
                })
                self.morph_plan.append({
                    'curvature': (k1_target, k2_target),
                    'stiffness': (0, 1),
                    'priority': 1
                })
            # Subcase 1.2: Co-direction change (e.g., both k1 and k2 up) -> Synchronize!
            else:
                print("INFO: Co-direction change. Planning synchronized morph.")
                delta_k1 = abs(k1_target - k1_current)
                delta_k2 = abs(k2_target - k2_current)

                # Find the smaller change, which will define the synchronized portion
                min_delta = min(delta_k1, delta_k2)
                
                # Phase 1: Both segments move by the smaller delta
                intermediate_k1 = round(float(k1_current + k1_dir * min_delta), 3)
                intermediate_k2 = round(float(k2_current + k2_dir * min_delta), 3)

                if delta_k1 < delta_k2:
                    self.morph_plan.append({
                        'curvature': (k1_target, intermediate_k2),
                        'stiffness': (1, 1),
                        'priority': 0
                    })
                    self.morph_plan.append({
                        'curvature': (k1_target, k2_target),
                        'stiffness': (0, 1),
                        'priority': 1
                    })
                else:
                    self.morph_plan.append({
                        'curvature': (intermediate_k1, k2_target),
                        'stiffness': (1, 1),
                        'priority': 1
                    })
                    self.morph_plan.append({
                        'curvature': (k1_target, k2_target),
                        'stiffness': (1, 0),
                        'priority': 0
                    })                        
        # For all other cases (no change, single change)
        else:
            # Create a simple, one-step plan
            self.morph_plan.append({
                'curvature': (k1_target, k2_target),
                'stiffness': (int(needs_k1_change), int(needs_k2_change)),
                'priority': 0
            })

    def _is_current_step_achieved(self) -> bool:
        """
        Checks if the robot's state matches the target for the current plan step.
        """        
        if self.morph_plan[0]['stiffness'] == (0, 0):
            return False
            
        current_stiffness_target = self.morph_plan[0]['curvature']

        k1_achieved = abs(self.robot.k1 - current_stiffness_target[0]) < self.k_threshold
        k2_achieved = abs(self.robot.k2 - current_stiffness_target[1]) < self.k_threshold

        achieved = [k1_achieved, k2_achieved]

        # Stop when the segment with the highest bending priority achieves its target curvature
        return achieved[self.morph_plan[0]['priority']]

class MMControl:
    """
    A state-driven controller for combined motion and morphology tasks.

    This class implements a high-level state machine that switches between
    different Model Predictive Control (MPC) configurations based on the robot's
    task. It can prioritize rigid body motion (changing x, y, theta) or shape
    morphing (changing stiffness parameters k1, k2).

    The unique properties of each control mode, including its kinematic equations,
    are defined in a dictionary. A single generic MPC engine is then configured 
    dynamically based on the active mode.

    Attributes:
        robot (robot_state.Model): The robot state object, containing current pose,
                                   curvature, and velocities.
        T (int): The number of time steps in the MPC prediction horizon.
        kinematics_handler (kinematics.HybridKinematics): Handles kinematic calculations.
        CONTROL_MODES (dict): A dictionary defining the properties of each
                              control mode.
    """
    def __init__(self, robot: robot_state.Model, T: int = 11, max_speed: float = 0.5, ema_alpha: float = 0.2):
        self.robot = robot
        self.__target_robot = copy.copy(robot)
        self.morph = Morphing(robot)
        self.kinematics_handler = kinematics.HybridKinematics()
        
        self.T = T  # MPC prediction horizon
        self.MAX_SPEED = max_speed

        # --- EMA Filter Parameters ---
        self.ema_alpha = ema_alpha
        # Stores the previous filtered velocity state. Initialized to zeros.
        self.__last_filtered_vel = np.zeros(5) 

        # This dictionary declares each MPC controller.
        # 'kinematics' is a function reference to the specific physics equations.
        # 'mpc' will hold the initialized GEKKO model instance.
        self.CONTROL_MODES = {
            'RIGID_MOTION': {
                'description': "Controls the robot's (x, y, theta) pose.",
                'kinematics': self._rigid_motion_kinematics,
                'mpc': None
            },
            'SHAPE_MORPH_1': {
                'description': "Changes k1 while holding k2 constant.",
                'kinematics': self._shape_morph_1_kinematics,
                'mpc': None
            },
            'SHAPE_MORPH_2': {
                'description': "Changes k2 while holding k1 constant.",
                'kinematics': self._shape_morph_2_kinematics,
                'mpc': None
            },
            'SHAPE_MORPH_3': {
                'description': "Changes k1 and k2 simultaneously.",
                'kinematics': self._shape_morph_3_kinematics,
                'mpc': None
            }
        }

        # Last used MPC model
        self.current_mpc = None  

        self._initialize_mpc_models()

    @property
    def target(self) -> robot_state.Model:
        return self.__target_robot
    
    @target.setter
    def target(self, value) -> None:
        if len(value) != 5:
            raise ValueError("Wrong number of target state values!")
        self.__target_robot.config = value

    def _initialize_mpc_models(self):
        """Creates an MPC instance for each mode defined in CONTROL_MODES."""
        for mode in self.CONTROL_MODES:
            self.CONTROL_MODES[mode]['mpc'] = self._generic_mpc_factory(mode)

    def _generic_mpc_factory(self, mode: str) -> GEKKO:
        """
        A factory function to create and configure a generic GEKKO MPC model.

        Args:
            mode (str): The name of the control mode, used to select the correct
                        kinematics.

        Returns:
            GEKKO: A configured GEKKO model instance ready for use.
        """
        m = GEKKO(remote=False)
        m.time = np.linspace(0, gv.DT * (self.T - 1), self.T)

        # --- Manipulated Variables (MVs) ---
        # These are the control inputs the optimizer can change.
        m.v_x = m.MV(value=0.0, lb=-0.09, ub=0.09)
        m.v_y = m.MV(value=0.0, lb=-0.09, ub=0.09)
        m.omega = m.MV(value=0.0, lb=-0.15, ub=0.15)
        m.u1 = m.MV(value=0.0, lb=-0.03, ub=0.03) # Velocity of k1
        m.u2 = m.MV(value=0.0, lb=-0.03, ub=0.03) # Velocity of k2

        # STATUS=1 tells the optimizer to adjust these variables
        m.v_x.STATUS = 1
        m.v_y.STATUS = 1
        m.omega.STATUS = 1
        m.u1.STATUS = 1
        m.u2.STATUS = 1

        # DCOST penalizes changes in the MV, encouraging smoother control
        m.v_x.DCOST = 0.05
        m.v_y.DCOST = 0.05
        m.omega.DCOST = 0.01
        m.u1.DCOST = 0.002
        m.u2.DCOST = 0.002

        # m.u1.DMAX = 0.005
        # m.u2.DMAX = 0.005

        # --- Controlled Variables (CVs) / State Variables ---
        # These are the system states we want to control
        m.x = m.CV(value=self.robot.x)
        m.y = m.CV(value=self.robot.y)
        m.theta = m.CV(value=self.robot.theta)
        m.k1 = m.CV(value=self.robot.k1)
        m.k2 = m.CV(value=self.robot.k2)

        # FSTATUS=1 enables feedback from measurements
        m.x.FSTATUS = 1
        m.y.FSTATUS = 1
        m.theta.FSTATUS = 1
        m.k1.FSTATUS = 1
        m.k2.FSTATUS = 1

        # STATUS=1 includes the CV in the objective function
        m.x.STATUS = 1
        m.y.STATUS = 1
        m.theta.STATUS = 1
        m.k1.STATUS = 1
        m.k2.STATUS = 1
        
        # CV_TYPE=2 uses a squared error objective: (CV - SP)^2
        m.options.CV_TYPE = 2

        # TR_INIT=1 enables a reference trajectory, making the setpoint (SP)
        # approach smoother instead of being a sudden step change
        m.x.TR_INIT = 1
        m.y.TR_INIT = 1
        m.theta.TR_INIT = 1
        m.k1.TR_INIT = 1
        m.k2.TR_INIT = 1

        # TAU is the time constant for the reference trajectory.
        # Larger TAU = slower, smoother approach to the setpoint.
        m.x.TAU = 3.2
        m.y.TAU = 3.2
        m.theta.TAU = 1.0
        m.k1.TAU = 5.0
        m.k2.TAU = 5.0

        # --- The Kinematic Model ---
        # This dynamically injects the specific physics equations for the
        # requested control mode into the generic MPC model.
        self.CONTROL_MODES[mode]['kinematics'](m)
        
        # --- Solver Settings ---
        m.options.IMODE = 6  # MPC Control Mode
        m.options.SOLVER = 3 # IPOPT solver

        return m
    
    # --- MPC Kinematics Builder Functions ---

    def _rigid_motion_kinematics(self, m: GEKKO):
        m.Equation(m.x.dt() == m.cos(m.theta) * m.v_x - m.sin(m.theta) * m.v_y)
        m.Equation(m.y.dt() == m.sin(m.theta) * m.v_x + m.cos(m.theta) * m.v_y)
        m.Equation(m.theta.dt() == m.omega)
        # Explicitly state that morphology does not change in this mode
        m.Equation(m.k1.dt() == 0)
        m.Equation(m.k2.dt() == 0)
        # Explicitly constrain the unused MVs to zero
        m.Equation(m.u1 == 0)
        m.Equation(m.u2 == 0)

    def _shape_morph_1_kinematics(self, m: GEKKO) -> list:
        m.Equation(m.v_x == 0)
        m.Equation(m.v_y == 0)
        m.Equation(m.omega == 0)

        k2_ratio = self.kinematics_handler.cardioid2.k_dot(self.robot.k1) / self.kinematics_handler.cardioid1.k_dot(self.robot.k1)
        pos = self.kinematics_handler.cardioid1.pos_dot(self.robot.theta, self.robot.k1, 1, 2)

        m.Equation(m.x.dt() == k2_ratio * pos[0] * m.u2)
        m.Equation(m.y.dt() == k2_ratio * pos[1] * m.u2)
        m.Equation(m.theta.dt() == self.kinematics_handler.cardioid2.th_dot(self.robot.k1) * m.u2)
        m.Equation(m.k1.dt() == -self.kinematics_handler.cardioid1.k_dot(self.robot.k1) * m.u1 + 
                   self.kinematics_handler.cardioid2.k_dot(self.robot.k1) * m.u2)
        m.Equation(m.k2.dt() == 0)

    def _shape_morph_2_kinematics(self, m: GEKKO):
        m.Equation(m.v_x == 0)
        m.Equation(m.v_y == 0)
        m.Equation(m.omega == 0)

        k1_ratio = self.kinematics_handler.cardioid2.k_dot(self.robot.k2) / self.kinematics_handler.cardioid1.k_dot(self.robot.k2)
        pos = self.kinematics_handler.cardioid1.pos_dot(self.robot.theta, self.robot.k2, 2, 1)
        
        m.Equation(m.x.dt() == k1_ratio * pos[0] * m.u1)
        m.Equation(m.y.dt() == k1_ratio * pos[1] * m.u1)
        m.Equation(m.theta.dt() == self.kinematics_handler.cardioid2.th_dot(self.robot.k2) * m.u1)
        m.Equation(m.k2.dt() == -self.kinematics_handler.cardioid2.k_dot(self.robot.k2) * m.u1 + 
                self.kinematics_handler.cardioid1.k_dot(self.robot.k2) * m.u2)
        m.Equation(m.k1.dt() == 0)

    def _shape_morph_3_kinematics(self, m: GEKKO):
        m.Equation(m.v_x == 0)
        m.Equation(m.v_y == 0)
        m.Equation(m.omega == 0)

        k1_ratio = self.kinematics_handler.cardioid3.k_dot(self.robot.k2) / self.kinematics_handler.cardioid1.k_dot(self.robot.k2)
        pos1 = self.kinematics_handler.cardioid1.pos_dot(self.robot.theta, self.robot.k2, 2, 1)

        k2_ratio = self.kinematics_handler.cardioid3.k_dot(self.robot.k1) / self.kinematics_handler.cardioid1.k_dot(self.robot.k1)
        pos2 = self.kinematics_handler.cardioid1.pos_dot(self.robot.theta, self.robot.k1, 1, 2)

        m.Equation(m.x.dt() == k1_ratio * pos1[0] * m.u1 + k2_ratio * pos2[0] * m.u2)
        m.Equation(m.y.dt() == k1_ratio * pos1[1] * m.u1 + k2_ratio * pos2[1] * m.u2)
        m.Equation(m.theta.dt() == self.kinematics_handler.cardioid3.th_dot(self.robot.k2) * m.u1 + 
                   self.kinematics_handler.cardioid3.th_dot(self.robot.k1) * m.u2)
        m.Equation(m.k1.dt() == -self.kinematics_handler.cardioid3.k_dot(self.robot.k1) * m.u1 + 
                   self.kinematics_handler.cardioid3.k_dot(self.robot.k1) * m.u2)
        m.Equation(m.k2.dt() == -self.kinematics_handler.cardioid3.k_dot(self.robot.k2) * m.u1 + 
                   self.kinematics_handler.cardioid3.k_dot(self.robot.k2) * m.u2)

    # --- Method to apply the EMA filter ---
    def _apply_ema_filter(self, raw_velocities: list) -> np.ndarray:
        """
        Applies an Exponential Moving Average filter to the raw velocity commands.
        
        Args:
            raw_velocities (list): The unfiltered velocity vector from the MPC.
        
        Returns:
            np.ndarray: The smoothed velocity vector.
        """
        # Convert raw list to a numpy array for vectorized math
        raw_velocities_np = np.array(raw_velocities)
        
        # Apply the EMA formula
        filtered_vel = (self.ema_alpha * raw_velocities_np) + \
                        ((1 - self.ema_alpha) * self.__last_filtered_vel)
        self.__last_filtered_vel = filtered_vel
        
        return filtered_vel
        
    
    def go_to_target(self) -> tuple:
        """
        Calculates the required velocities to reach a target configuration.

        This is the main entry point for the controller. It acts as a state
        machine: MORPHING -> MOVING -> IDLE.

        Args:
            target_config (list): The desired final state of the robot as
                                  [x, y, theta, k1, k2].

        Returns:
            A tuple containing:
            - is_finished (bool): True if the robot has reached its target.
            - velocities (list): The calculated optimal [vx, vy, w, u1, u2].
            - stiff_transitions (list): Commands for the stiffness actuators.
            - q_new (np.array): The predicted next state of the robot.
        """
        # 1. Determine the current control mode based on the target.
        current_mode, current_k_target, stiff_transitions = self.morph.update_control_mode(self.target.curvature)
        is_finished = (current_mode == 'RIGID_MOTION' and self._is_pose_close(self.target.position, dist_thresh=0.001))

        # --- DEBUGGING PRINTOUT ---
        plan_list = list(self.morph.morph_plan)
        print(f"\nMorph Plan  : {plan_list}")
        print("Mode        : " + current_mode)

        if is_finished:
            self.morph.morph_plan.clear()
        
        # 2. If idle or finished, command zero velocity.
        if current_mode == 'IDLE' or is_finished:
            raw_velocities  = [0.0] * 5
            filtered_velocities = [0.0] * 5
            # The robot's configuration doesn't change.
            q_new = self.robot.config
        else:
            # 3. If a mode is active, prepare and solve the MPC problem.
            mode_mpc_model = self.CONTROL_MODES[current_mode]['mpc']

            # If the MPC model has changed, update the CV initial values
            if self.current_mpc != mode_mpc_model:
                mode_mpc_model.x.VALUE = self.robot.x
                mode_mpc_model.y.VALUE = self.robot.y
                mode_mpc_model.theta.VALUE = self.robot.theta
                mode_mpc_model.k1.VALUE = self.robot.k1
                mode_mpc_model.k2.VALUE = self.robot.k2

                self.current_mpc = mode_mpc_model

            # --- Initialize the MPC state ---

            # Provide the current state measurements as feedback.
            mode_mpc_model.x.MEAS = self.robot.x
            mode_mpc_model.y.MEAS = self.robot.y
            mode_mpc_model.theta.MEAS = self.robot.theta
            mode_mpc_model.k1.MEAS = self.robot.k1
            mode_mpc_model.k2.MEAS = self.robot.k2

            # Set the desired final state (setpoints).
            mode_mpc_model.x.SP = self.target.x
            mode_mpc_model.y.SP = self.target.y
            mode_mpc_model.theta.SP = self.target.theta
            mode_mpc_model.k1.SP = current_k_target[0]
            mode_mpc_model.k2.SP = current_k_target[1]

            # 4. Solve for the optimal control action.
            mode_mpc_model.solve(disp=False)

            # 5. Extract the optimal velocities for the current time step.
            raw_velocities  = [
                mode_mpc_model.v_x.NEWVAL, mode_mpc_model.v_y.NEWVAL,
                mode_mpc_model.omega.NEWVAL, mode_mpc_model.u1.NEWVAL,
                mode_mpc_model.u2.NEWVAL
            ]

            raw_velocities = [round(v, 5) for v in raw_velocities]

            # 6. Filter the raw optimal velocities
            filtered_velocities = self._apply_ema_filter(raw_velocities)

            # 7. Update the MPC model's MV values for the next cycle's start.
            mode_mpc_model.v_x.VALUE = filtered_velocities[0]
            mode_mpc_model.v_y.VALUE = filtered_velocities[1]
            mode_mpc_model.omega.VALUE = filtered_velocities[2]
            mode_mpc_model.u1.VALUE = filtered_velocities[3]
            mode_mpc_model.u2.VALUE = filtered_velocities[4]
            
            # Predict the next state using the calculated velocities.
            # q_new = [
            #     mode_mpc_model.x.PRED[1], mode_mpc_model.y.PRED[1],
            #     mode_mpc_model.theta.PRED[1], mode_mpc_model.k1.PRED[1],
            #     mode_mpc_model.k2.PRED[1]
            # ]
            q_new = self.robot.config + self.kinematics_handler.get_unified_jacobian(
                self.robot, plan_list[0]['stiffness']
            ).dot(filtered_velocities) * gv.DT
            q_new = q_new.tolist()
  
        return raw_velocities, filtered_velocities, stiff_transitions, q_new, is_finished
    
    def _is_pose_close(self, target_pos: list, dist_thresh: float = 0.018) -> bool:
        """
        Checks if the robot's planar position is within a threshold of the target.
        """
        dist = np.linalg.norm(np.array(self.robot.position) - np.array(target_pos))
        print(f'Pos error   : {dist}')
        return dist < dist_thresh
