import numpy as np
from gekko import GEKKO
from entities import global_var as gv, robot_state
from control import stiffness
import kinematics


class MotionMorphologyControl:
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
        stiffness_controller (stiffness.FSM): A finite state machine to manage
                                              transitions between stiffness states.
        kinematics_handler (kinematics.HybridKinematics): Handles kinematic calculations.
        CONTROL_MODES (dict): A dictionary defining the properties of each
                              control mode.
    """
    def __init__(self, robot: robot_state.Model, T: int = 21, max_speed: float = 0.5):
        self.robot = robot
        self.T = T  # MPC prediction horizon
        self.MAX_SPEED = max_speed

        self.stiffness_controller = stiffness.FSM(robot)
        self.kinematics_handler = kinematics.HybridKinematics()

        # This dictionary maps target stiffness states to control mode names.
        self.STIFFNESS_TO_MODES = {
            (0, 0): 'RIGID_MOTION',
            (1, 0): 'SHAPE_MORPH_1',
            (0, 1): 'SHAPE_MORPH_2',
            (1, 1): 'SHAPE_MORPH_3'
        }

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

        self._initialize_mpc_models()

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
        m.v_y = m.MV(value=0.0, lb=-0.07, ub=0.07)
        m.omega = m.MV(value=0.0, lb=-0.7, ub=0.7)
        m.u_1 = m.MV(value=0.0, lb=-0.05, ub=0.05) # Velocity of k1
        m.u_2 = m.MV(value=0.0, lb=-0.05, ub=0.05) # Velocity of k2

        # STATUS=1 tells the optimizer to adjust these variables
        m.v_x.STATUS = 1
        m.v_y.STATUS = 1
        m.omega.STATUS = 1
        m.u_1.STATUS = 1
        m.u_2.STATUS = 1

        # DCOST penalizes changes in the MV, encouraging smoother control
        m.v_x.DCOST = 1.0
        m.v_y.DCOST = 1.0
        m.omega.DCOST = 0.5
        m.u_1.DCOST = 1.0
        m.u_2.DCOST = 1.0

        # --- Controlled Variables (CVs) / State Variables ---
        # These are the system states we want to control
        m.x = m.CV()
        m.y = m.CV()
        m.theta = m.CV()
        m.k1 = m.CV()
        m.k2 = m.CV()

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
        m.x.TAU = 1.0
        m.y.TAU = 1.0
        m.theta.TAU = 1.2
        m.k1.TAU = 1.0
        m.k2.TAU = 1.0

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
        m.Equation(m.u_1 == 0)
        m.Equation(m.u_2 == 0)

    def _shape_morph_1_kinematics(self, m: GEKKO) -> list:
        pass

    def _shape_morph_2_kinematics(self, m: GEKKO):
        # ... Port logic from original mpcSM2 ...
        pass

    def _shape_morph_3_kinematics(self, m: GEKKO):
        # ... Port logic from original mpcSM3 ...
        pass

    # --- State Logic and Main Control Loop ---
    
    def _determine_morph_mode(self, stiff_config: list) -> tuple:
        """
        Determines the required control mode based on the target stiffness.

        Args:
            stiff_config (list): The target [k1, k2] values.

        Returns:
            A tuple containing the determined mode (str), stiffness transition
            commands, and the target stiffness tuple.
        """
        k1_diff = abs(self.robot.k1 - stiff_config[0])
        k2_diff = abs(self.robot.k2 - stiff_config[1])
        k_threshold = 4.0  # Threshold to decide if a stiffness change is needed.

        stiff1 = 1 if k1_diff > k_threshold else 0
        stiff2 = 1 if k2_diff > k_threshold else 0
        target_stiffness = (stiff1, stiff2)

        # Use the FSM to handle transitions and debounce signals.
        is_transitioning, stiff_transitions = self.stiffness_controller.main(target_stiffness)
        
        current_mode = 'IDLE' if is_transitioning else self.STIFFNESS_TO_MODES.get(target_stiffness)
        
        return current_mode, target_stiffness, stiff_transitions

    def go_to_target(self, target_config: list) -> tuple:
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
        current_mode, target_stiffness, stiff_transitions = self._determine_morph_mode(target_config[3:])
        is_finished = (current_mode == 'RIGID_MOTION' and self._is_pose_close(target_config[:2]))

        # 2. If idle or finished, command zero velocity.
        if current_mode == 'IDLE' or is_finished:
            velocities = [0.0] * 5
            # The robot's configuration doesn't change.
            q_new = self.robot.config
        else:
            # 3. If a mode is active, prepare and solve the MPC problem.
            mode_mpc_model = self.CONTROL_MODES[current_mode]['mpc']

            # --- Initialize the MPC state ---

            # Provide the current state measurements as feedback.
            mode_mpc_model.x.MEAS = self.robot.x
            mode_mpc_model.y.MEAS = self.robot.y
            mode_mpc_model.theta.MEAS = self.robot.theta
            mode_mpc_model.k1.MEAS = self.robot.k1
            mode_mpc_model.k2.MEAS = self.robot.k2

            # Set the desired final state (setpoints).
            mode_mpc_model.x.SP = target_config[0]
            mode_mpc_model.y.SP = target_config[1]
            mode_mpc_model.theta.SP = target_config[2]
            mode_mpc_model.k1.SP = target_config[3]
            mode_mpc_model.k2.SP = target_config[4]

            # 4. Solve for the optimal control action.
            mode_mpc_model.solve(disp=False)

            # 5. Extract the optimal velocities for the current time step.
            velocities = [
                mode_mpc_model.v_x.NEWVAL, mode_mpc_model.v_y.NEWVAL,
                mode_mpc_model.omega.NEWVAL, mode_mpc_model.u_1.NEWVAL,
                mode_mpc_model.u_2.NEWVAL
            ]

            # 6. Set the optimal velocities as the starting point for the MVs 
            # during the next cycle.
            mode_mpc_model.v_x.VALUE = velocities[0]
            mode_mpc_model.v_y.VALUE = velocities[1]
            mode_mpc_model.omega.VALUE = velocities[2]
            mode_mpc_model.u_1.VALUE = velocities[3]
            mode_mpc_model.u_2.VALUE = velocities[4]
            
            # Predict the next state using the calculated velocities.
            q_new = self.robot.config + self.kinematics_handler.get_unified_jacobian(
                self.robot, target_stiffness
            ).dot(velocities) * gv.DT
            
        return velocities, stiff_transitions, q_new, is_finished
    
    def _is_pose_close(self, target_pos: list, dist_thresh: float = 0.018) -> bool:
        """
        Checks if the robot's planar position is within a threshold of the target.
        """
        dist = np.linalg.norm(np.array(self.robot.position) - np.array(target_pos))
        return dist < dist_thresh
