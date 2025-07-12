"""
This module provides the kinematic calculations for a 2SR robot, a mobile
platform with a deformable body composed of two segments of variable stiffness.
It implements the unified Jacobian that governs the robot's motion in both
its rigid and flexible states.
"""

import numpy as np
from typing import List
from entities import global_var as gv, robot_state

class _Cardioid:
    """
    An internal helper class to model the path of a locomotion unit (LU) during 
    body flexion.

    When a robot segment bends, the LU at its end traces a predictable path
    that is accurately modeled by a cardioid. This class encapsulates the
    geometric properties of these paths. The robot's curvature (kappa) exhibits
    an inverse linear relationship with the cardioid's rolling angle (phi),
    allowing for precise kinematic mapping.
    """
    def __init__(self, n: int):
        """
        Initializes the Cardioid model with parameters from global constants.

        Args:
            n (int): The identifier for the cardioid scenario (1, 2, or 3),
                     used to look up its specific geometric parameters (radius, angle range).
        """
        self.a = gv.CARDIOID_A[n-1]
        self.phi_min = gv.CARDIOID_TH_MIN[n-1]
        self.phi_max = gv.CARDIOID_TH_MAX[n-1]
        self.offset = gv.CARDIOID_OFFSET[n-1]

        # Pre-calculate constants for the linear mapping between curvature and phi.
        self.delta_phi = self.phi_max - self.phi_min
        self.var_phi = np.pi / (gv.L_VSS * self.delta_phi)

    # Map segment curvature 'kappa' to the cardioid's rolling angle 'phi'
    def phi(self, kappa: float) -> float:
        return self.phi_min + (1 / self.var_phi) * (np.pi / (2 * gv.L_VSS) - kappa)

    # Calculate the radius 'rho' of the cardioid for a given curvature 'kappa'
    def rho(self, kappa: float) -> float:
        phi = self.phi(kappa)
        return 2 * self.a * (1 - np.cos(phi))

    # Calculate the derivative of the robot's position (body frame) w.r.t. curvature
    def pos_dot(self, th0: float, kappa: float, seg: int, lu: int) -> List[float]:
        phi = self.phi(kappa)
        th = th0 + (-1)**seg * kappa * gv.L_VSS
        pos_dot_matrix = np.array([
            [2 * self.a * (-np.sin(phi + th) + np.sin(2 * phi + th))],
            [(-1)**(lu - 1) * 2 * self.a * (np.cos(phi + th) - np.cos(2 * th))]
        ])
        return pos_dot_matrix.flatten().tolist()

    # Calculate the derivative of the robot's orientation
    def th_dot(self, kappa: float) -> float:
        return self.k_dot(kappa) * gv.L_VSS

    # Calculate the derivative of curvature
    def k_dot(self, kappa: float) -> float:
        return self.var_phi / self.rho(kappa)


class HybridKinematics:
    """
    Calculates the unified hybrid Jacobian for the 2SR robot.

    This class implements the core kinematic model, providing a single matrix
    J(q, s) that maps a unified control vector u = [u_r, u_s] to the rate of
    change of the robot's configuration, q_dot.
    """
    def __init__(self):
        """
        Initializes the kinematics engine by pre-instantiating the three
        cardioid models used for different flexible states.
        """
        self.cardioid1 = _Cardioid(1)
        self.cardioid2 = _Cardioid(2)
        self.cardioid3 = _Cardioid(3)

        # --- Kinematic Mode Rules ---
        # This dictionary declares the configuration for each stiffness state (s1, s2).
        # It specifies which cardioid models to use and whether the rigid
        # component of the Jacobian is active.
        self.KINEMATIC_MODES = {
            # Key: tuple(s_flags) -> (s1, s2)
            # Value: A dictionary defining the mode's configuration.
            (0, 0): {
                "is_rigid_active": True,
                "spiral1_model": self.cardioid1, # Default, not used
                "spiral2_model": self.cardioid2, # Default, not used
            },
            (0, 1): {
                "is_rigid_active": False,
                "spiral1_model": self.cardioid1,
                "spiral2_model": self.cardioid2,
            },
            (1, 0): {
                "is_rigid_active": False,
                "spiral1_model": self.cardioid1,
                "spiral2_model": self.cardioid2,
            },
            (1, 1): {
                "is_rigid_active": False,
                "spiral1_model": self.cardioid3, # Both flexible, use Cardioid 3
                "spiral2_model": self.cardioid3,
            },
        }

    def get_unified_jacobian(self, robot: robot_state.Model, s_flags: List[int]) -> np.ndarray:
        """
        Constructs the unified Jacobian J = [J_r, J_s].

        This method acts as the "engine". It looks up the configuration for the
        current stiffness state from KINEMATIC_MODES and uses it to assemble
        the final Jacobian matrix.

        Args:
            robot (robot_state.Model): The robot's current state object.
            s_flags (List[int]): The stiffness configuration [s1, s2], where a value
                                 of 0 indicates a rigid segment and 1 a flexible one.

        Returns:
            np.ndarray: The 5x5 unified hybrid Jacobian matrix, J.
        """

        # 1. Look up the declarative rule for the current stiffness mode.
        mode_rule = self.KINEMATIC_MODES[tuple(s_flags)]

        # 2. Compute the base Jacobian components. These are pure math functions.
        J_rigid = self._get_rigid_jacobian_component(robot)
        J_flexible = self._get_flexible_jacobian_component(
            robot,
            s_flags,
            spiral1_model=mode_rule["spiral1_model"],
            spiral2_model=mode_rule["spiral2_model"]
        )

        # 3. Use the rule to determine if the rigid component is active.
        is_rigid_active = mode_rule["is_rigid_active"]
        active_J_rigid = int(is_rigid_active) * J_rigid

        # 4. Assemble the final Jacobian based on the rule.
        return np.hstack((active_J_rigid, J_flexible))

    def _get_rigid_jacobian_component(self, robot: robot_state.Model) -> np.ndarray:
        """
        Computes the rigid-body Jacobian, J_r.

        This 5x3 matrix maps the robot's body velocity u_r = [vx, vy, omega]
        to configuration changes (q_dot) when the robot is in its rigid state
        (s = [0, 0]). Curvatures (kappa_1, kappa_2) are constant in this mode.
        """
        theta = robot.theta
        return np.array([
            [np.cos(theta), -np.sin(theta), 0],  # Maps [vx, vy] to [x_dot, y_dot]
            [np.sin(theta),  np.cos(theta), 0],
            [0,              0,             1],  # Maps omega to theta_dot
            [0,              0,             0],  # kappa_1 is constant
            [0,              0,             0]   # kappa_2 is constant
        ])

    def _get_flexible_jacobian_component(self, robot: robot_state.Model, s_flags: List[int],
                                         spiral1_model: _Cardioid, spiral2_model: _Cardioid) -> np.ndarray:
        """
        Computes the "soft" Jacobian for flexible states, J_s.

        This 5x2 matrix maps actuator velocities u_s = [v1, v2] to configuration
        changes when one or both segments are flexible (s != [0, 0]). It is
        calculated as the Hadamard product of a stiffness-based selector matrix
        and a matrix of kinematic coupling terms derived from the cardioid models.
        """
        k1_ratio = spiral2_model.k_dot(robot.k2) / self.cardioid1.k_dot(robot.k2)
        k2_ratio = spiral2_model.k_dot(robot.k1) / self.cardioid1.k_dot(robot.k1)
        pos_lu1 = self.cardioid1.pos_dot(robot.theta, robot.k2, 2, 1)
        pos_lu2 = self.cardioid1.pos_dot(robot.theta, robot.k1, 1, 2)

        J_flex_unscaled = np.array([
            [k1_ratio * pos_lu1[0], k2_ratio * pos_lu2[0]],
            [k1_ratio * pos_lu1[1], k2_ratio * pos_lu2[1]],
            [spiral2_model.th_dot(robot.k2), spiral2_model.th_dot(robot.k1)],
            [-spiral1_model.k_dot(robot.k1), spiral2_model.k_dot(robot.k1)],
            [-spiral2_model.k_dot(robot.k2), spiral1_model.k_dot(robot.k2)]
        ])

        # This matrix acts as a selector based on the stiffness flags [s1, s2].
        s1, s2 = s_flags
        stiffness_selector = np.array([
            [s2, s1], 
            [s2, s1], 
            [s2, s1],
            [s1, s1],
            [s2, s2]
        ])

        # Compute the final J_s via the Hadamard (element-wise) product.
        return np.multiply(stiffness_selector, J_flex_unscaled)