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
        # This will hold a sequence of morphing commands (target curvatures and stiffness).
        self.morph_plan = collections.deque()
        self.k_threshold = 0.1  # Curvature tolerance.

    def update_control_mode(self, target_k_config: list) -> tuple:
        """
        Determines the next step in a morphing plan.
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
        # Determine the current motion & deformation mode
        current_mode = self.STIFFNESS_TO_MODES.get(current_stiffness_target)

        # If any VSS is in the process of heating or cooling switch the mode to 'IDLE'
        # to let the robot wait for the stiffness transition to complete
        if is_transitioning:
            current_mode = 'IDLE'
            
        return current_mode, current_k_target, stiff_transitions

    def _create_morph_plan(self, target_k_config: list):
        """
        Analyzes the target and creates a sequential plan of stiffness states.
        """
        k1_target, k2_target = target_k_config[0], target_k_config[1]

        # Determine if a change is needed for each segment
        needs_k1_change = abs(self.robot.k1 - k1_target) > self.k_threshold
        needs_k2_change = abs(self.robot.k2 - k2_target) > self.k_threshold
        
        # Determine the direction of change (1 for increase, -1 for decrease, 0 for no change)
        k1_dir = int(np.sign(k1_target - self.robot.k1)) if needs_k1_change else 0
        k2_dir = int(np.sign(k2_target - self.robot.k2)) if needs_k2_change else 0
        
        # If both need to change in different directions
        if needs_k1_change and needs_k2_change and k1_dir != k2_dir:
            print("INFO: Split-direction change. Planning sequential morph.")
            # Plan: Morph k1, then k2.
            self.morph_plan.append({
                'curvature': (k1_target, self.robot.k2),
                'stiffness': (1, 0)
            })
            self.morph_plan.append({
                'curvature': (k1_target, k2_target),
                'stiffness': (0, 1)
            })        
        # For all other cases create a simple, one-step plan
        else:
            self.morph_plan.append({
                'curvature': (k1_target, k2_target),
                'stiffness': (int(needs_k1_change), int(needs_k2_change))
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

        if self.morph_plan[0]['stiffness'][0] and k1_achieved or self.morph_plan[0]['stiffness'][1] and k2_achieved:
            return True
        else:
            return False

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
    def __init__(self, robot: robot_state.Model, T: int = 11, max_speed: float = 0.5):
        self.robot = robot
        self.__target_robot = copy.copy(robot)
        self.morph = Morphing(robot)
        self.kinematics_handler = kinematics.HybridKinematics()
        
        self.T = T  # MPC prediction horizon
        self.MAX_SPEED = max_speed

        # Desired speed of the pose change
        self.desired_speed = [0.07, 0.07, 0.2]
        self.tau_array = [0.0] * 3
        self.tau_scale = 0.98

        # This dictionary declares each MPC controller.
        # 'kinematics' is a function reference to the specific mode equations.
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

        self.current_mpc = None # Last used MPC model

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
        m.omega = m.MV(value=0.0, lb=-0.3, ub=0.3)
        m.u1 = m.MV(value=0.0, lb=-0.03, ub=0.03) # Velocity of k1
        m.u2 = m.MV(value=0.0, lb=-0.03, ub=0.03) # Velocity of k2

        # STATUS=1 tells the optimizer to adjust these variables
        m.v_x.STATUS = 1
        m.v_y.STATUS = 1
        m.omega.STATUS = 1
        m.u1.STATUS = 1
        m.u2.STATUS = 1

        # DCOST penalizes changes in the MV, encouraging smoother control
        m.v_x.DCOST = 1
        m.v_y.DCOST = 1
        m.omega.DCOST = 1
        m.u1.DCOST = 1
        m.u2.DCOST = 1

        # --- Controlled Variables (CVs) / State Variables ---
        # These are the system states to control
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
        # TAU values for the pose are set dynamically via update_tau() 
        # every time the new target is set
        m.x.TAU = 1.0
        m.y.TAU = 1.0
        m.theta.TAU = 1.0
        m.k1.TAU = 4.0
        m.k2.TAU = 4.0

        # --- The Kinematic Model ---
        # This dynamically injects the specific kinematics equations for the
        # requested control mode into the generic MPC model.
        self.CONTROL_MODES[mode]['kinematics'](m)
        
        # --- Solver Settings ---
        m.options.IMODE = 6  # MPC Control Mode

        # m.options.SOLVER = 3 # IPOPT solver
        m.options.SOLVER = 2 # NOTE: use a BPOPT solver if IPOPT doesn't work

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
         
    def update_tau(self):
        """
        Updates the MPC trajectory time constants based on the difference between
        the initial and target poses.
        """
        for i in range(len(self.robot.pose)):
            self.tau_array[i] = float(abs(self.target.pose[i] - self.robot.pose[i]) / self.desired_speed[i])

        pose_tau = max(self.tau_array[:3])

        self.tau_array[0] = pose_tau
        self.tau_array[1] = pose_tau
        self.tau_array[2] = 0.65 * pose_tau
        
        for mode in self.CONTROL_MODES:
            mpc = self.CONTROL_MODES[mode]['mpc']

            mpc.x.TAU = self.tau_array[0]
            mpc.y.TAU = self.tau_array[1]
            mpc.theta.TAU = self.tau_array[2]      

            self.CONTROL_MODES[mode]['mpc'] = mpc
    
    def go_to_target(self) -> tuple:
        """
        Calculates the required velocities to reach a target configuration.
        This is the main entry point for the controller. 

        Returns:
            A tuple containing:
            - velocities (list): The calculated optimal [vx, vy, w, u1, u2].
            - stiff_transitions (list): Commands for the stiffness actuators.
            - q_new (np.array): The predicted next state of the robot.
            - is_finished (bool): True if the robot has reached its target.
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
            self.current_mpc = None

            for mode in self.CONTROL_MODES:
                self.CONTROL_MODES[mode]['mpc'] = None
                
            self._initialize_mpc_models()
        
        # 2. If idle or finished, command zero velocity.
        if current_mode == 'IDLE' or is_finished:
            optimal_velocities  = [0.0] * 5
            # The robot's configuration doesn't change.
            q_new = self.robot.config.tolist()
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

            # Scale TAU for faster approach.
            mode_mpc_model.x.TAU *= self.tau_scale
            mode_mpc_model.y.TAU *= self.tau_scale
            mode_mpc_model.theta.TAU *= self.tau_scale
            mode_mpc_model.k1.TAU *= self.tau_scale
            mode_mpc_model.k2.TAU *= self.tau_scale

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
            optimal_velocities  = [
                mode_mpc_model.v_x.NEWVAL, mode_mpc_model.v_y.NEWVAL,
                mode_mpc_model.omega.NEWVAL, mode_mpc_model.u1.NEWVAL,
                mode_mpc_model.u2.NEWVAL
            ]

            # 6. Update the MPC model's MV values for the next cycle's start.
            mode_mpc_model.v_x.VALUE = optimal_velocities[0]
            mode_mpc_model.v_y.VALUE = optimal_velocities[1]
            mode_mpc_model.omega.VALUE = optimal_velocities[2]
            mode_mpc_model.u1.VALUE = optimal_velocities[3]
            mode_mpc_model.u2.VALUE = optimal_velocities[4]

            self.CONTROL_MODES[current_mode]['mpc']  = mode_mpc_model
            
            # 7. Predict the next state using the calculated velocities.
            q_new = self.robot.config + self.kinematics_handler.get_unified_jacobian(
                self.robot, plan_list[0]['stiffness']
            ).dot(optimal_velocities) * gv.DT
            q_new = q_new.tolist()
  
        return optimal_velocities, stiff_transitions, q_new, is_finished
    
    def _is_pose_close(self, target_pos: list, dist_thresh: float = 0.018) -> bool:
        """
        Checks if the robot's planar position is within a threshold of the target.
        """
        dist = np.linalg.norm(np.array(self.robot.position) - np.array(target_pos))
        print(f'Pos error   : {dist}')
        return dist < dist_thresh
