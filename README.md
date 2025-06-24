# 2SR Agent

**2SRA (Self-Reconfigurable Roft-Rigid Agent)** is a compact mobile robot that introduces a novel approach to mobile manipulation and navigation. By integrating variable stiffness capabilities, 2SRA can switch between a rigid, omnidirectional mobile platform and a flexible, deformable manipulator. This unique design enables two key functionalities:
* **Full-Body Grasping:** In its flexible state, 2SRA can envelop and secure objects by conforming its body to their shape. This method of grasping removes the dependency of mobile platforms on dedicated manipulators like arms or grippers, thereby streamlining the robot's design, and reducing both its size and mechanical complexity. After grasping, the robot can revert to a rigid state to transport the object effectively.
* **Morphology-Aware Navigation:** 2SRA leverages its adaptive morphology to efficiently navigate its environment. When faced with narrow passages or cluttered areas, the robot can reconfigure its shape to squeeze through obstacles that would block a conventional, rigid robot. This capability is a form of *embodied intelligence*, where the physical structure of the robot is an integral part of its navigation strategy.

By unifying the functions of a robust mobile platform and a deformable manipulator, 2SRA represents a *versatile, minimalist robot* offering a compelling solution for tasks requiring both mobility and dexterity in environments where size and complexity are critical constraints.

<!-- Link to the paper -->

## Hardware Architecture

The 2SRA platform is built with a *modular* design philosophy, separating mobility from reconfigurability. It consists of two Locomotion Units connected by the novel Variable-Stiffness Bridge.

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

Stiffness Model

FSM control (animation)

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
