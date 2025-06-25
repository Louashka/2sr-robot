# 2SR Agent

**2SRA (Self-Reconfigurable Roft-Rigid Agent)** is a compact mobile robot that introduces a novel approach to mobile manipulation and navigation. By integrating variable stiffness capabilities, 2SRA can switch between a rigid, omnidirectional mobile platform and a flexible, deformable manipulator. This unique design enables two key functionalities:
* **Full-Body Grasping:** In its flexible state, 2SRA can envelop and secure objects by conforming its body to their shape. This method of grasping removes the dependency of mobile platforms on dedicated manipulators like arms or grippers, thereby streamlining the robot's design, and reducing both its size and mechanical complexity. After grasping, the robot can revert to a rigid state to transport the object effectively.
* **Morphology-Aware Navigation:** 2SRA leverages its adaptive morphology to efficiently navigate its environment. When faced with narrow passages or cluttered areas, the robot can reconfigure its shape to squeeze through obstacles that would block a conventional, rigid robot. This capability is a form of *embodied intelligence*, where the physical structure of the robot is an integral part of its navigation strategy.

By unifying the functions of a robust mobile platform and a deformable manipulator, 2SRA represents a *versatile, minimalist robot* offering a compelling solution for tasks requiring both mobility and dexterity in environments where size and complexity are critical constraints.

<!-- Link to the paper -->

## Hardware Architecture

The 2SRA platform is built with a *modular* design philosophy, separating mobility from reconfigurability. It consists of two Locomotion Units connected by the novel Variable-Stiffness Bridge.

![Hardware Design Diagram](images/design.svg)

### Locomotion Units (LUs)

The robot's mobility is provided by two self-contained, wheeled units that house all the electronics, batteries, and motors required for autonomous operation. They act as the control base of the robot.

### Modular Variable-Stiffness Bridge (VSB)

The key innovation of 2SRA is its bridge, which enables the robot to dynamically switch between rigid and flexible states. This is the second iteration of our design, evolving from a monolithic to a fully modular system.

This new bridge is built on a cable chain backbone populated with *compact Variable-Stiffness Modules*. The principle of altering stiffness within these modules was adopted from the work of [Tonazzini et al](https://doi.org/10.1002/adma.201602580). The integration of a Low-Melting-Point Alloy (LMPA) and a coiled heater in a silicone shell allows to achieve two distinct states:
* **Rigid State**: When the alloy is cool and solid, the bridge is rigid, and the robot moves like a conventional three-DoF mobile platform.
* **Flexible State**: By heating the modules, the alloy melts, making the bridge flexible and deformable. This allows the robot to bend, conform to objects, and squeeze through tight spaces.

The modules are organized into two segments that can be actuated independently for a greater variety of bending shapes. This modular approach is a significant improvement over the original monolithic design, offering faster phase transitions (>10x improvement), lower power consumption (~1.7V per module), uniform bending, and enhanced reliability, as individual modules can be easily replaced. 

Detailed information about the first-generation design **(2SRA v1)** can be found [here](https://doi.org/10.1109/LRA.2023.3241749).

<!-- Desribe the desig, insert the image, later the animation

Links to the design, cad files, pcb, etc... -->

## Stiffness Control

To control the robot's ability to switch between rigid and soft states, we developed a system based on a **Finite-State Machine (FSM)**. This system manages the transitions for each of the two segments in the Variable-Stiffness Bridge.

The stiffness configuration of the robot is represented by a simple boolean vector: $\mathbf{s} = [s_1, s_2]^\intercal$, where $s_i$ is the state of the $i-$th segment:
* $0$: Rigid state (alloy is solid)
* $1$: Flexible state (alloy is liquid)

This allows for four unique stiffness configurations: $[0, 0]^\intercal$ (VSB is entirely rigid), $[1, 0]^\intercal$, $[0, 1]^\intercal$, $[1, 1]^\intercal$ (VSB is entirely flexible). Based on the desired state, the controller can issue one of three actions to each segment:
* $\;\;\:0$: maintain the current state
* $\;\;\:1$: initiate the alloy's melting (transition to flexible)
* $-1$: initiate the alloys cooling (transition to rigid)

### Handling Thermal Hysteresis

In this particular implementation, we use Field's, an alloy with a melting point of $\approx 62^\circ\text{C}$. Ideally, the alloy would melt at this temperature and solidify just below it. However, our system approximates the stiffness of an entire segment using a single temperature sensor. 

>This practical simplification, combined with the thermal dynamics of cooling, creates a **hysteresis loop**: the segment doesn't become rigid again at the same temperature it became flexible.

To address this, we include *two different temperature thresholds* in the FSM controller:
1. **Upper threshold** ($62^\circ\text{C}$): Confirms the segment is fully Flexible.
2. **Lower threshold** ($53^\circ\text{C}$): Confirms the segment is fully Rigid.

This two-threshold system makes the state transitions robust and reliable, preventing the robot from attempting to move before its structure is truly rigid. The animation below illustrates this process: a command is sent, the temperature changes, and the FSM waits for the correct threshold to be crossed before officially changing the segment's state. 

The complete logic for handling stiffness transitions in VSB segments is implemented in the [FSMController](control/stiffness_handler.py#L12) class.

## Hybrid Kinematics

Motion Modes

Cardioids

Control strategy

## Full-body Grasping

Describe the Method

## Mobile Manipulation

...

## Morphology-aware Navigation

...
