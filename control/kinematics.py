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
        self._cardioid1 = _Cardioid(1)
        self._cardioid2 = _Cardioid(2)
        self._cardioid3 = _Cardioid(3)

    def _get_rigid_jacobian(self, robot: robot_state.Model) -> np.ndarray:
        """
        (Private) Computes the rigid-body Jacobian, J_r.

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

    def _get_flexible_jacobian(self, robot: robot_state.Model, s_flags: List[int]) -> np.ndarray:
        """
        (Private) Computes the "soft" Jacobian for flexible states, J_s.

        This 5x2 matrix maps actuator velocities u_s = [v1, v2] to configuration
        changes when one or both segments are flexible (s != [0, 0]). It is
        calculated as the Hadamard product of a stiffness-based selector matrix
        and a matrix of kinematic coupling terms derived from the cardioid models.

        Args:
            robot (robot: robot_state.Model): The current state of the robot.
            s_flags (List[int]): The stiffness configuration [s1, s2], where 1 is
                                 flexible and 0 is rigid.

        Returns:
            np.ndarray: The 5x2 flexible Jacobian matrix (J_s).
        """
        # Select the appropriate cardioid models based on the flexible state.
        # If s = [1, 1], both segments are flexible, use Cardioid 3.
        # Otherwise, a mix of Cardioid 1 and 2 is used.
        if all(flag == 1 for flag in s_flags):
            spiral1 = spiral2 = self._cardioid3
        else:
            spiral1 = self._cardioid1
            spiral2 = self._cardioid2

        # Calculate kinematic coupling ratios and position derivatives.
        k1_ratio = spiral2.k_dot(robot.k2) / self._cardioid1.k_dot(robot.k2)
        k2_ratio = spiral2.k_dot(robot.k1) / self._cardioid1.k_dot(robot.k1)
        pos_lu1 = self._cardioid1.pos_dot(robot.theta, robot.k2, 2, 1)
        pos_lu2 = self._cardioid1.pos_dot(robot.theta, robot.k1, 1, 2)

        # Assemble the unscaled kinematic coupling matrix.
        J_flex_unscaled = np.array([
            [k1_ratio * pos_lu1[0], k2_ratio * pos_lu2[0]], # Corresponds to J_1n, J_2n
            [k1_ratio * pos_lu1[1], k2_ratio * pos_lu2[1]],
            [spiral2.th_dot(robot.k2), spiral2.th_dot(robot.k1)], # Corresponds to l*K_n terms
            [-spiral1.k_dot(robot.k1), spiral2.k_dot(robot.k1)], # Corresponds to -K_m, K_n terms
            [-spiral2.k_dot(robot.k2), spiral1.k_dot(robot.k2)]
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

    def get_unified_jacobian(self, robot: robot_state.Model, s_flags: List[int]) -> np.ndarray:
        """
        Constructs the unified Jacobian J = [J_r, J_s].

        This is the main public method. It combines the rigid and flexible Jacobians
        into a single 5x5 matrix. This matrix acts as a mode selector, dynamically
        adjusting how the unified control input u = [vx, vy, omega, v1, v2] maps
        to motion based on the stiffness configuration `s_flags`.

        Args:
            robot (robot_state.Model): The robot's current state object.
            s_flags (List[int]): The stiffness configuration [s1, s2], where a value
                                 of 0 indicates a rigid segment and 1 a flexible one.

        Returns:
            np.ndarray: The 5x5 unified hybrid Jacobian matrix, J.
        """
        # The robot is in rigid mode only if both stiffness flags are 0.
        is_rigid_mode = not any(flag == 1 for flag in s_flags)

        # Compute both Jacobian components.
        J_rigid = self._get_rigid_jacobian(robot)
        J_flexible = self._get_flexible_jacobian(robot, s_flags)

        # The rigid Jacobian component is zeroed out unless in pure rigid mode.
        # This correctly implements the `overline(s1 or s2)` logic from the formula.
        active_J_rigid = int(is_rigid_mode) * J_rigid

        # Horizontally stack the components to form the final unified Jacobian.
        # This creates the structure J = [J_r, J_s].
        return np.hstack((active_J_rigid, J_flexible))