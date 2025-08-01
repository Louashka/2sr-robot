import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
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

def plot_potential_on_contour(shape: manipulandum.Shape, min_potentials):
    """
    Calculates and plots the 'best grasp potential' for each point on the shape's contour.
    """
    # Get the contour points corresponding to s_values
    contour_points = np.array([shape.get_point(s) for s in s_values])
    x = contour_points[:, 0]
    y = contour_points[:, 1]

    # Create a set of line segments from the contour points
    # Segments are defined by a start and end point: [[(x1, y1), (x2, y2)], ...]
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(10, 9))

    # Create a LineCollection object, which can color each segment individually
    lc = LineCollection(segments, cmap='viridis', norm=plt.Normalize(0, np.max(min_potentials)))
    
    # Set the values used for colormapping. We use the potential at the start of each segment.
    lc.set_array(min_potentials)
    lc.set_linewidth(5)
    
    # Add the LineCollection to the axes
    line = ax.add_collection(lc)
    
    # Add a color bar to serve as a legend for the potential values
    cbar = fig.colorbar(line, ax=ax)
    cbar.set_label('Best Grasp Potential (0 = Best)', rotation=270, labelpad=20)

    # Plot the Center of Mass
    com = shape.centroid
    ax.plot(com[0], com[1], 'wx', markersize=12, mew=3, label='Center of Mass')
    ax.plot(com[0], com[1], 'kx', markersize=12, mew=1.5) # Black cross inside white for visibility

    # Formatting the plot
    ax.set_title('Grasp Potential Mapped on Object Contour')
    ax.set_xlabel('X Coordinate (m)')
    ax.set_ylabel('Y Coordinate (m)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_aspect('equal', adjustable='box') # Crucial for correct shape aspect ratio
    plt.show()

if __name__ == "__main__":
    object_pose = [0.5, -0.7, 3 * np.pi/4]
    object = manipulandum.Shape(object_id, object_pose)
    object.heading_local = -np.pi / 2
    contour = object.contour

    manip_controller = mobile_manipulation.ObjectHandlingPolicy()
    grasp_idx, approach, pre_grasp, final_contact = manip_controller.grasp(object)

    approach_model = robot_state.Model(1, *approach)
    pre_grasp_model = robot_state.Model(2, *pre_grasp)
    final_contact_model = robot_state.Model(3, *final_contact)

    contact_point = object.get_point(grasp_idx)


    resolution = 100
    print(f"Calculating potential field on contour with resolution {resolution}...")
    
    # s_values = np.linspace(0, 1, resolution)
    # min_potentials = np.zeros(resolution)

    # # For each point s1, find the best partner s2 that minimizes the potential
    # for i, s1 in enumerate(s_values):
    #     best_potential_for_s1 = float('inf')
    #     for s2 in s_values:
    #         potential = manip_controller.calculate_force_closure_potential(object, s1, s2)
    #         if potential < best_potential_for_s1:
    #             best_potential_for_s1 = potential
    #     min_potentials[i] = best_potential_for_s1

    # print("Calculation complete. Generating plot...")

    # plot_potential_on_contour(object, min_potentials)

    # Create a grid of s1 and s2 parameters
    s_values = np.linspace(0, 1, resolution)
    potential_grid = np.zeros((resolution, resolution))
    
    # Iterate over every pair of (s1, s2) to calculate the potential
    for i, s1 in enumerate(s_values):
        for j, s2 in enumerate(s_values):
            potential_grid[i, j] = manip_controller.calculate_force_closure_potential(object, s1, s2)



    fig, ax = plt.subplots(figsize=(8, 8))
    # robot_plotter = plotlib.RobotPlot(ax, display_temp=False)

    # ax.set_xlim((0.25, 0.65))
    # ax.set_ylim((-0.9, -0.5))
    ax.set_aspect('equal')

    param_contour = object.parametric_contour() 

    # ax.plot(contour[:,0], contour[:,1], '.r')
    # ax.plot(object.centroid[0], object.centroid[1], '*r')
    # ax.plot(param_contour[:,0], param_contour[:,1])
    # robot_plotter.plot_robot(final_contact_model)

    # We use 'T' to transpose the grid so s1 is on the x-axis and s2 is on the y-axis
    im = plt.imshow(potential_grid.T, origin='lower', extent=[0, 1, 0, 1], cmap='viridis', aspect='auto')
    
    plt.colorbar(im, label='Potential U (0 = Best, 2 = Worst)')
    plt.xlabel('Parameter s1 for Contact Point P1')
    plt.ylabel('Parameter s2 for Contact Point P2')
    plt.title(f'Force Closure Attraction Potential Landscape for a "{type(object).__name__}" Object')
    plt.grid(False)
    
    # Find and mark the global minimum
    min_val = np.min(potential_grid)
    min_indices = np.where(potential_grid == min_val)
    min_s1 = s_values[min_indices[0][0]]
    min_s2 = s_values[min_indices[1][0]]
    plt.plot(min_s1, min_s2, 'r*', markersize=15, label=f'Global Minimum at ({min_s1:.2f}, {min_s2:.2f})')
    
    plt.legend()

    plt.title(f"Object Contour (ID: {object_id})")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()