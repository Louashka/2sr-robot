import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from entities import manipulandum, robot_state
from control import mobile_manipulation
import plotlib

# object_id = 11 # Cheescake
# object_id = 12 # Ellipse
object_id = 13 # Heart
# object_id = 14 # Bean

if __name__ == "__main__":
    object_pose = [0.5, -0.7, np.pi/4]
    object = manipulandum.Shape(object_id, object_pose)
    object.heading_local = np.pi / 2
    contour = object.contour

    manip_controller = mobile_manipulation.ObjectHandlingPolicy()
    grasp_idx, approach, pre_grasp, final_contact = manip_controller.grasp(object)

    approach_model = robot_state.Model(1, *approach)
    pre_grasp_model = robot_state.Model(2, *pre_grasp)
    final_contact_model = robot_state.Model(3, *final_contact)

    contact_point = object.get_point(grasp_idx)

    fig, ax = plt.subplots(figsize=(8, 8))
    robot_plotter = plotlib.RobotPlot(ax, display_temp=False)

    # ax.set_xlim((-0.2, 0.15))
    # ax.set_ylim((-0.2, 0.15))
    ax.set_aspect('equal')

    param_contour = object.parametric_contour() 

    center = object.centroid(object.contour)
    param_center = object.centroid(param_contour)

    ax.plot(contour[:,0], contour[:,1], '.r')
    ax.plot(center[0], center[1], '*r')
    ax.plot(object.x, object.y, 'o')
    ax.plot(param_contour[:,0], param_contour[:,1])
    ax.plot(param_center[0], param_center[1], '*k')
    ax.plot(contact_point[0], contact_point[1], "*k")
    robot_plotter.plot_robot(final_contact_model)

    plt.title(f"Object Contour (ID: {object_id})")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()