# 2SR Robot

**2SR (Self-Reconfigurable Soft-Rigid)** is a compact mobile robot that introduces a novel approach to mobile manipulation and navigation. By integrating variable stiffness capabilities, 2SR can switch between a rigid, omnidirectional mobile platform and a flexible, deformable manipulator. This unique design enables two key functionalities:
* **Full-Body Grasping:** In its flexible state, 2SR can envelop and secure objects by conforming its body to their shape. This method of grasping removes the dependency of mobile platforms on dedicated manipulators like arms or grippers, thereby streamlining the robot's design, and reducing both its size and mechanical complexity. After grasping, the robot can revert to a rigid state to transport the object effectively.
* **Morphology-Aware Navigation:** 2SR leverages its adaptive morphology to efficiently navigate its environment. When faced with narrow passages or cluttered areas, the robot can reconfigure its shape to squeeze through obstacles that would block a conventional, rigid robot. This capability is a form of *embodied intelligence*, where the physical structure of the robot is an integral part of its navigation strategy.

By unifying the functions of a robust mobile platform and a deformable manipulator, 2SR represents a *versatile, minimalist robot* offering a compelling solution for tasks requiring both mobility and dexterity in environments where size and complexity are critical constraints.

<!-- Link to the paper -->

## Hardware Architecture

The 2SR platform is built with a *modular* design philosophy, separating mobility from reconfigurability. It consists of two Locomotion Units connected by the novel Variable-Stiffness Bridge.

![Hardware Design Diagram](images/design.svg)

### Locomotion Units (LUs)

The robot's mobility is provided by two self-contained, wheeled units that house all the electronics, batteries, and motors required for autonomous operation. They act as the control base of the robot.

### Modular Variable-Stiffness Bridge (VSB)

The key innovation of 2SR is its bridge, which enables the robot to dynamically switch between rigid and flexible states. This is the second iteration of our design, evolving from a monolithic to a fully modular system.

This new bridge is built on a cable chain backbone populated with *compact Variable-Stiffness Modules*. The principle of altering stiffness within these modules was adopted from the work of [Tonazzini et al](https://doi.org/10.1002/adma.201602580). The integration of a Low-Melting-Point Alloy (LMPA) and a coiled heater in a silicone shell allows to achieve two distinct states:
* **Rigid State**: When the alloy is cool and solid, the bridge is rigid, and the robot moves like a conventional three-DoF mobile platform.
* **Flexible State**: By heating the modules, the alloy melts, making the bridge flexible and deformable. This allows the robot to bend, conform to objects, and squeeze through tight spaces.

The modules are organized into two segments (VSS) that can be actuated independently for a greater variety of bending shapes. This modular approach is a significant improvement over the original monolithic design, offering faster phase transitions (>10x improvement), lower power consumption (~1.7V per module), uniform bending, and enhanced reliability, as individual modules can be easily replaced. 

Detailed information about the first-generation design **(2SR v1)** can be found [here](https://doi.org/10.1109/LRA.2023.3241749).

<!-- Desribe the desig, insert the image, later the animation

Links to the design, cad files, pcb, etc... -->

## Stiffness Control

To control the robot's ability to switch between rigid and soft states, we developed a system based on a **Finite-State Machine (FSM)**. This system manages the transitions for each of the two segments in the Variable-Stiffness Bridge.

The stiffness configuration of the robot is represented by a simple boolean vector: $\mathbf{s} = [s_1, s_2]^\intercal$, where $s_i$ is the stiffness state of the $i\text{-th}$ segment. Based on the current and desired states, the controller can issue one of three actions to each segment: 
* $0$: maintain the current state
* $1$: initiate the alloy's melting (transition to flexible)
* $-1$: initiate the alloys cooling (transition to rigid). 

The complete state-transition table of the FSM stiffness controller is provided below.

| State | Input | Next State | Output |
| :---: | :---: | :---: | :---: |
| 0 (rigid) | 0 | 0 (rigid) | None |
| 0 (rigid) | 1 | 1 (flexible) | Melts LMPA $\rightarrow$ VSS turns flexible |
| 0 (rigid) | -1 | 0 (rigid) | None |
| 1 (flexible) | 0 | 1 (flexible) | None |
| 1 (flexible) | 1 | 1 (flexible) | None |
| 1 (flexible) | -1 | 0 (rigid) | Solidifies LMPA $\rightarrow$ VSS turns rigid |

### Handling Thermal Hysteresis

In this particular implementation, we use Field's metal, an alloy with a melting point of $\approx62^\circ\text{C}$. Ideally, the alloy would melt at this temperature and solidify just below it. However, our system approximates the stiffness of an entire segment using a single temperature sensor. This practical simplification, combined with the thermal dynamics of cooling, creates a **hysteresis loop**: the segment doesn't become rigid again at the same temperature it became flexible.

To address this, we include *two different temperature thresholds* in the FSM controller:
1. **Upper threshold** ($62^\circ\text{C}$): Confirms the segment is fully Flexible.
2. **Lower threshold** ($53^\circ\text{C}$): Confirms the segment is fully Rigid.

This two-threshold system makes the state transitions robust and reliable, preventing the robot from attempting to move before its structure is truly rigid. The animation below illustrates this process: a command is sent, the temperature changes, and the FSM waits for the correct threshold to be crossed before officially changing the segment's state. 

The complete logic for handling stiffness transitions in VSB segments is implemented in the [FSMController](control/stiffness_handler.py#L12) class.

## Hybrid Kinematics

### Robot's Configuration

Since the robot's shape is not fixed, we define its complete configuration using five generalized coordinates: 
$$\mathbf{q} = [x, y, \theta, \kappa_1, \kappa_2]^\intercal,$$
where
* $(x, y)$ is the robot's global position
* $\theta$ is the robot's orientation
* $(\kappa_1, \kappa_2)$ is the curvature of the first and second segments, respectively. These values define the robot's shape.

### Operation Modes

#### 1. Rigid State ($\mathbf{s} = [0, 0]^\intercal$)

When both segments are rigid, the robot's shape is locked ($\kappa_1$ and $\kappa_2$ are constant). In this mode, it behaves like a standard omnidirectional mobile platform:
```math
\dot{\textbf{q}} = \underbrace{\overline{(s_1 \vee s_2)}
    \begin{bmatrix}
        \textbf R_z\left(\theta\right) \\
        0 
     \end{bmatrix}}_{\mathbf{J}_r}\textbf u_r,
```
where $\textbf R_z\left(\theta\right) \in \mathbb{R}^3$ is a rotation matrix around the vertical axis of the global frame and $\textbf u_r = [v_{x}, v_{y}, \omega]^\intercal$ is a vector of robot's "rigid" control velocities.   

#### 2. Flexible States ($\mathbf{s} \in \{[1, 0]^\intercal, [0, 1]^\intercal, [1, 1]^\intercal\}$)

When one or both segments are flexible, the kinematics become far more complex. The wheels no longer just drive the robot; they also actively bend the body. The key insight was discovering that as a segment bends, the wheel at its end traces a predictable path that can be accurately modeled by a **cardioid**. 

Depending on which segment is flexible and which wheel is moving, the robot follows *one of three distinct cardioid trajectories* to control its shape:
1.  One segment is flexible while the other remains rigid. The LU adjacent to the flexible segment is in motion (Cardioid 1)
2. One segment is flexible while the other remains rigid. The LU adjacent to the rigid segment is in motion (Cardioid 2)
3. Both segments are flexible, with either LU moving (Cardioid 3)

Through path analysis and curve fitting, we determine the cardioid's radius $r$ and rolling angle $\phi$ range for each scenario, as listed in the table below.

| Cardioid | $r$ | $\phi^{\text{min}}$ | $\phi^{\text{max}}$ |
| :---: | :---: | :---: | :---:|
| 1 | 0.021 | 2.42 | 3.87 |
| 2 | 0.049 | 2.19 | 4.09 |
| 3 | 0.043 | 1.73 | 4.56 |

The VSS curvature exhibits an inverse linear relationship with the rolling angle $\phi$, enabling reliable tracking of the robot's frame displacement using the cardioid equations. The robot's motion in flexible states is controlled through "soft" velocities $\mathbf{u}_s = [v_1, v_2]^T$, where $v_j$ represents the velocity of the $j$-th locomotion unit traversing a specific cardioid path. Based on the geometry of these cardioids, we derived the following "soft" Jacobian:
```math
\textbf J_{s} = 
        \begin{bmatrix}
            s_2 & s_1 \\
            s_2 & s_1 \\
            s_2 & s_1 \\
            s_1 & s_1 \\
            s_2 & s_2 
        \end{bmatrix} \circ
        \begin{bmatrix}
           \mathbf J_{1n}(\kappa_2,\theta) &  \mathbf J_{2n}(\kappa_1,\theta) \\
           lK_n(\kappa_2) & lK_n(\kappa_1) \\
           -K_m(\kappa_1) & K_n(\kappa_1) \\
           -K_n(\kappa_2) & K_m(\kappa_2) \\
        \end{bmatrix}
```

### The Unified Control Framework

To manage the complex behaviour of the 2SR robot, we developed a comprehensive control strategy:

1. **Unified Jacobian:** A single, unified Jacobian matrix combining both operational modes acts as a "mode selector." It dynamically adjusts how wheel velocities map to robot motion (both position and shape) based on the current stiffness configuration. This allows one mathematical model to govern all possible states:
```math
\begin{gathered}
\dot{\mathbf{q}} = \mathbf{J}(\mathbf{q}, \mathbf{s})\mathbf{u}\\
         \mathbf{J}(\mathbf{q}, \mathbf{s}) = [\mathbf{J}_r,\mathbf{J}_s]^\intercal, \quad \mathbf{u} = [\mathbf{u}_r,\mathbf{u}_s]^\intercal
\end{gathered}
```
2. **Model Predictive Control (MPC):** With the kinematics defined, we use Model Predictive Control to generate the precise wheel velocities needed to reach a target configuration. There are four separate MPC controllers, one for each stiffness state. The system activates the appropriate controller for the current mode.
3. **Supervisory Controller:** High-level logic in Motion & Morphology (M&M) Controller decides when to change stiffness versus when to just move. It optimizes for efficiency by keeping the robot rigid by default and only activating a shape change when necessary.

## Full-body Grasping

Describe the Method

## Mobile Manipulation

...

## Morphology-aware Navigation

...
