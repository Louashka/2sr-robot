# =============================================================================
#
# flexible_robot_analysis.py
#
# DESCRIPTION:
#   A script for collecting and analyzing the behavior of a 2SR robot
#   under different control inputs. It can be configured to run for
#   various stiffness modes (e.g., semi-flexible, fully flexible).
#
#   The script performs two main tasks:
#   1.  collect_data: Simulates the robot's movement for a grid of control
#       inputs (v1, v2) and saves the high-resolution trajectory data to a
#       compressed Parquet file.
#   2.  analyse_data: Loads the saved data and generates two key figures:
#       - A set of 2D heatmaps showing performance metrics across the control space.
#       - A set of 3D trajectory plots visualizing how each state variable
#         evolves over time for every control input pair.
#
# USAGE:
#   1.  Set up the desried robot stiffness (robot.stiff1 and robot.stiff2).
#   2.  Set the CURRENT_MODE variable to the desired mode you want to run.
#   3.  To generate new data, uncomment the `collect_data(robot, mode_config)` line.
#   4.  To analyze existing data, run the script with the `analyse_data(mode_config)` line active.
#
# REQUIREMENTS:
#   - pandas
#   - pyarrow
#   - fastparquet
#   - matplotlib
#
# =============================================================================

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# --- Add project root to Python path for custom module imports ---
# This allows the script to find modules like 'entities' and 'kinematics'
# when run from the 'analysis' directory.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from entities import global_var as gv, robot_state
import kinematics, plotlib

# =============================================================================
# SCRIPT CONFIGURATION
# =============================================================================

VARS_COLORS = {
    "x": "viridis",
    "y": "viridis",
    "theta": "plasma",
    "k1": "cool",
    "k2": "cool",
}

# This dictionary defines the parameters for each flexible operational mode of 
# the robot.
MODES = {
    (1, 0): {
        'file_path': "analysis/data/semi_flex_1_data.parquet",
        'hm_fig_path': "analysis/figures/semi_flex_1_hm.pdf",
        'traj_fig_path': "analysis/figures/semi_flex_1_traj.pdf",
        'state_vars': ['x', 'y', 'theta', 'k1'],
        'curvature_to_display': "k1",
        'title': 'Semi-Flex Mode 1 (k1 deformable)'
    },
    (0, 1): {
        'file_path': "analysis/data/semi_flex_2_data.parquet",
        'hm_fig_path': "analysis/figures/semi_flex_2_hm.pdf",
        'traj_fig_path': "analysis/figures/semi_flex_2_traj.pdf",
        'state_vars': ['x', 'y', 'theta', 'k2'],
        'curvature_to_display': "k2",
        'azim': -180,
        'title': 'Semi-Flex Mode 2 (k2 deformable)'
    },
    (1, 1): {
        'file_path': "analysis/data/flex_data.parquet",
        'hm_fig_path': "analysis/figures/flex_hm.pdf",
        'traj_fig_path': "analysis/figures/flex_traj.pdf",
        'state_vars': ['x', 'y', 'theta', 'k1'],
        'curvature_to_display': "k1",
        'title': 'Fully Flexible Mode (k1, k2 deformable)'
    }
}

def collect_data(robot: robot_state.Model, config: dict):
    """
    Runs simulations for a grid of control inputs and saves the resulting
    trajectory data to a Parquet file.

    Args:
        robot (robot_state.Model): The robot model instance to be simulated.
        config (dict): The configuration dictionary for the current mode from MODES.
    """
    # --- Simulation Parameters ---
    dt = 0.01  # Simulation time step
    k_max = np.pi / (2 * gv.L_VSS)  # Maximum curvature before termination
    init_state = robot.config.tolist()  # Store initial state to reset after each run
    file_path = config['file_path']

    # --- Define Control Input Space ---
    # Creates a grid of 100x100 control inputs for v1 and v2
    v_neg = np.linspace(-gv.LU_SPEED, 0, 50, endpoint=False)
    v_pos = np.linspace(0, gv.LU_SPEED, 50)
    v1_array = v2_array = v_neg.tolist() + v_pos.tolist()

    # --- Data Collection ---
    performance_records = []
    kinematics_handler = kinematics.HybridKinematics()

    for v1 in v1_array:
        for v2 in v2_array:
            if v1 == 0.0 and v2 == 0.0:
                continue

            print(f"Calculating for ({v1:.3f}, {v2:.3f})...")
            step_counter = 0
            while True:
                # --- Simulation Step ---
                # Use the mapping from the config to create the command vector
                command_vector = [0.0, 0.0, 0.0, v1, v2]
                J = kinematics_handler.get_unified_jacobian(robot, robot.stiffness)
                q_dot = J.dot(command_vector)
                new_config = robot.config + q_dot * dt
                robot.config = new_config.tolist()

                # --- Data Storage ---
                # Create a dictionary (a "record") for the current time step
                current_record = {
                    'v1': v1,
                    'v2': v2,
                    'time': step_counter * dt,
                    'x': robot.config[0],
                    'y': robot.config[1],
                    'theta': robot.config[2],
                    'k1': robot.config[3],
                    'k2': robot.config[4]
                }
                performance_records.append(current_record)
                step_counter += 1

                # --- Termination Check ---
                # Stop if curvature or rotation limits are exceeded
                if (abs(robot.k1) >= k_max or abs(robot.k2) >= k_max or 
                    abs(robot.theta - init_state[2]) >= 2 * np.pi):
                    robot.config = init_state  # Reset robot for next run
                    break

    # --- Save Data ---
    print("\n...Creating DataFrame...")
    df_performance = pd.DataFrame(performance_records)
    
    # Ensure the directory exists before saving
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    print(f"Saving performance data to {file_path}...")
    df_performance.to_parquet(file_path, engine='pyarrow', compression='snappy')
    print("\nData collection complete!")


def analyse_data(config: dict):
    """
    Loads and analyzes trajectory data, generating and displaying summary plots.

    Args:
        config (dict): The configuration dictionary for the current mode from MODES.
    """
    file_path = config['file_path']
    print(f"Loading data from {file_path}...")
    if not os.path.exists(file_path):
        print(f"Error: Data file not found at {file_path}.")
        print("Please run the `collect_data` function first.")
        return

    df = pd.read_parquet(file_path)
    var_scale = 100 # Scaling factor for visualization (e.g., m to cm)

    # plot_workspace_trajectories(df, config, var_scale)
    # plot_3d(df)

    # --- Generate and Display Plots ---
    hm_fig = plot_heatmaps(df, config, var_scale)
    # traj_fig = plot_3d_trajectories(df, config, var_scale)

    # print("\nSaving figures...")
    hm_fig.savefig(config["hm_fig_path"], dpi=300, transparent=True)
    # traj_fig.savefig(config["traj_fig_path"], dpi=300, transparent=True)

    print("\nDisplaying generated figures...")
    plt.show()

def plot_workspace_trajectories(df: pd.DataFrame, config: dict, var_scale: float):
    """
    Generates a two-part figure to visualize the robot's workspace.

    The left subplot shows all trajectories to define the reachable workspace.
    The right subplot uses the RobotPlot class to draw the robot's final
    shape for a representative subset of control inputs, illustrating a "sweep"
    of its possible configurations.

    Args:
        df (pd.DataFrame): The input dataframe containing all trajectory data.
        config (dict): The configuration for the current mode.
        var_scale (float): A scaling factor for display units (e.g., 100 for m to cm).
    """
    print("Generating 2D workspace and robot shape sweep plots...")

    # --- Plotting Setup ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 8))
    fig.suptitle(f'Workspace Analysis ({config["title"]})', fontsize=16)

    # --- Subplot 1: Full Reachable Workspace ---
    print("  Plotting all trajectories for reachable workspace...")
    ax1.set_title('Reachable Workspace')
    ax1.set_xlabel('x position [cm]')
    ax1.set_ylabel('y position [cm]')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_aspect('equal', adjustable='box')

    # Plot every trajectory with a light, transparent color.
    # The density of lines will show the most common paths.
    for _, traj in df.groupby(["v1", "v2"]):
        ax1.plot(
            traj['x'] * var_scale,
            traj['y'] * var_scale,
            color='#333333', # A dark gray color
            alpha=0.05,     # Very transparent
            linewidth=1.0
        )

    # --- Subplot 2: Robot Shape Sweep ---
    print("  Plotting a sweep of final robot shapes...")
    ax2.set_title('Final Robot Poses (Subset Sweep)')
    ax2.set_xlabel('x position [cm]')
    ax2.set_ylabel('y position [cm]')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.set_aspect('equal', adjustable='box')

    # Instantiate the provided robot plotter on the second axes
    robot_plotter = plotlib.RobotPlot(ax=ax2)

    # Get the final state (last row) of each trajectory
    final_states = df.groupby(["v1", "v2"]).last()

    # To avoid a completely black plot, we only draw the robot for a subset
    # of the final states. Let's plot ~100 robot shapes.
    num_trajectories = len(final_states)
    subsample_rate = max(1, num_trajectories // 100)

    for i, (_, row) in enumerate(final_states.iterrows()):
        if i % subsample_rate == 0:
            # Create a robot_state instance from the final state data in the row.
            # Note: We scale x and y to cm here so the plot is in the correct units.
            # The RobotPlot class expects its internal calculations to be in meters,
            # but by passing scaled x/y, the final drawing lands on the correct
            # coordinates of our cm-based axes.
            robot_instance = robot_state.Model(
                id = 2,
                x=row['x'] * var_scale,
                y=row['y'] * var_scale,
                theta=row['theta'],
                k1=row['k1'],
                k2=row['k2']
            )
            # The RobotPlot class uses its own internal lengths (gv.L_VSS etc)
            # which are in meters. We need to scale them for the cm plot.
            # A cleaner way is to let RobotPlot draw in meters and just set the axes limits.
            # Let's revert to the unscaled version and set limits later.
            robot_instance_meters = robot_state.Model(1, row.x, row.y, row.theta, row.k1, row.k2)
            robot_plotter.plot_robot(robot_instance_meters)

    # Ensure the scales of both plots match for easy comparison.
    # We get the limits from the first plot (in cm) and apply them to the second.
    xlim = ax1.get_xlim()
    ylim = ax1.get_ylim()
    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

def plot_3d(df: pd.DataFrame):
    sub_df = df.iloc[::10]
    fig = plt.figure(figsize=(14, 6))

    # First subplot: z = x
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.scatter(sub_df['v1'], sub_df['v2'], sub_df['x'], c=sub_df['x'], cmap='viridis', alpha=0.7)
    ax1.set_xlabel('v1')
    ax1.set_ylabel('v2')
    ax1.set_zlabel('x')
    ax1.set_title('3D Plot: v1, v2, x')

    # Second subplot: z = y
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.scatter(sub_df['v1'], sub_df['v2'], sub_df['y'], c=sub_df['y'], cmap='plasma', alpha=0.7)
    ax2.set_xlabel('v1')
    ax2.set_ylabel('v2')
    ax2.set_zlabel('y')
    ax2.set_title('3D Plot: v1, v2, y')

    plt.tight_layout()


def plot_heatmaps(df: pd.DataFrame, config: dict, var_scale: float):
    """
    Generates a 2x2 grid of heatmaps for key performance metrics.
    - Top Row: Absolute max displacement in X and Y.
    - Bottom Row: Total rotation and final curvature.
    """
    print("\nGenerating control space performance heatmaps...")
    
    curv_to_display = config.get("curvature_to_display", "k1")

    summary_df = df.groupby(["v1", "v2"]).agg(
        max_abs_x=('x', lambda s: s.abs().max()),
        max_abs_y=('y', lambda s: s.abs().max()),
        total_rotation=('theta', lambda t: abs(t.iloc[-1] - t.iloc[0])),
        final_curvature=(curv_to_display, 'last')
    ).reset_index()

    pivot_x = summary_df.pivot(index='v2', columns='v1', values='max_abs_x') * var_scale
    pivot_y = summary_df.pivot(index='v2', columns='v1', values='max_abs_y') * var_scale
    pivot_rot = summary_df.pivot(index='v2', columns='v1', values='total_rotation')
    pivot_curv = summary_df.pivot(index='v2', columns='v1', values='final_curvature')

    fig, axes = plt.subplots(2, 2, figsize=(15, 13))
    fig.suptitle(f'Control Space Performance Metrics ({config["title"]})', fontsize=16)

    v1_min, v1_max = df['v1'].min() * var_scale, df['v1'].max() * var_scale
    v2_min, v2_max = df['v2'].min() * var_scale, df['v2'].max() * var_scale
    plot_extent = [v1_min, v1_max, v2_min, v2_max]
    
    # TOP ROW: DISPLACEMENT (Shared 'viridis' colormap)
    vmax_disp = max(pivot_x.max().max(), pivot_y.max().max())
    
    ax = axes[0, 0]
    im = ax.imshow(pivot_x.values, extent=plot_extent, cmap='viridis', aspect='equal', origin='lower', vmin=0, vmax=vmax_disp)
    # ax.set_title('Absolute Max Displacement in X')
    ax.set_ylabel(r'$v_2$ [cm/s]')
    fig.colorbar(im, ax=ax, label='Max x Displacement [cm]')

    ax = axes[0, 1]
    im = ax.imshow(pivot_y.values, extent=plot_extent, cmap='viridis', aspect='equal', origin='lower', vmin=0, vmax=vmax_disp)
    # ax.set_title('Absolute Max Displacement in Y')
    fig.colorbar(im, ax=ax, label='Max y Displacement [cm]')

    # BOTTOM ROW: ROTATION & CURVATURE (Independent colormaps)
    ax = axes[1, 0]
    im = ax.imshow(pivot_rot.values, extent=plot_extent, cmap='plasma', aspect='equal', origin='lower')
    # ax.set_title('Total Rotation')
    ax.set_xlabel(r'$v_1$ [cm/s]'); ax.set_ylabel(r'$v_2$ [cm/s]')
    fig.colorbar(im, ax=ax, label='Total Rotation [rad]')

    ax = axes[1, 1]
    im = ax.imshow(pivot_curv.values, extent=plot_extent, cmap='coolwarm', aspect='equal', origin='lower')
    # ax.set_title(f'Final Curvature ({curv_to_display})')
    ax.set_xlabel(r'$v_1$ [cm/s]')
    fig.colorbar(im, ax=ax, label=f'Final Curvature ($m^{{-1}}$)')

    axes[0, 1].set_yticklabels([])
    axes[0, 0].set_xticklabels([])
    axes[0, 1].set_xticklabels([])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    return fig


def plot_3d_trajectories(df: pd.DataFrame, config: dict, var_scale: float):
    """
    Generates 3D plots showing the evolution of each state variable over the
    entire control space.

    Args:
        df (pd.DataFrame): The input dataframe containing all trajectory data.
        config (dict): The configuration for the current mode.
        var_scale (float): A scaling factor for display units.
    """
    print("\nGenerating 3D trajectory plots...")
    
    # Subsample the data for faster plotting. Plotting every single point is slow
    # and often not necessary for visualization.
    df_sub = df[::10].copy() # Take every 10th row

    state_vars_to_plot = config['state_vars']
    num_plots = len(state_vars_to_plot)
    fig = plt.figure(figsize=(12, 5 * (num_plots // 2)))
    # fig.suptitle(f'State Trajectories over Control Space ({config["title"]})', fontsize=18)

    # --- Create a subplot for each state variable ---
    for i, state_var in enumerate(state_vars_to_plot):
        ax = fig.add_subplot( (num_plots + 1) // 2, 2, i + 1, projection='3d')

        # Normalize colors based on the final value of the state for each trajectory
        final_values = df_sub.groupby(['v1', 'v2'])[state_var].last()
        norm = Normalize(vmin=final_values.min(), vmax=final_values.max())
        cmap = plt.get_cmap(VARS_COLORS[state_var])

        # Group by each control input pair and plot its trajectory as a line
        for (v1, v2), group in df_sub.groupby(['v1', 'v2']):
            final_val = final_values.loc[(v1, v2)]
            color = cmap(norm(final_val))
            
            # The z-axis is the evolving state variable
            z_values = group[state_var]
            if state_var in ['x', 'y']:
                z_values *= var_scale # Scale position variables to cm

            ax.plot(group['v1'] * var_scale, group['v2'] * var_scale, z_values,
                    color=color, alpha=0.5, linewidth=0.8)

        # --- Formatting ---
        z_label = f'{state_var} [cm]' if state_var in ['x', 'y'] else state_var
        if state_var == 'theta': z_label = r'$\theta$ [rad]'
        if state_var == 'k1': z_label = r'$k_1$ [$m^{-1}$]'
        if state_var == 'k2': z_label = r'$k_2$ [$m^{-1}$]'
        
        # ax.set_title(f'Evolution of {state_var}')
        ax.set_xlabel(r'$v_1$ [cm/s]', labelpad=10)
        ax.set_ylabel(r'$v_2$ [cm/s]', labelpad=10)
        ax.set_zlabel(z_label, labelpad=10)
        # ax.view_init(azim=-145)

    plt.tight_layout()

    return fig


# =============================================================================
# MAIN EXECUTION BLOCK
# =============================================================================
if __name__ == "__main__":
    # --- Step 1: Initialize the robot model ---
    init_state = tuple([0.0] * 5)
    robot = robot_state.Model(1, *init_state)

    # --- Step 2: Set up desired stiffness ---
    robot.stiff1 = 1 
    robot.stiff2 = 1

    # --- Step 3: Determine the operational mode ---
    current_mode = tuple(robot.stiffness)
    mode_config = MODES[current_mode]

    # --- Step 4: Choose an action ---
    # To generate new data, uncomment the line below.
    # NOTE: This can take a long time to run.

    # collect_data(robot, mode_config)

    # To analyze existing data, run the line below.
    analyse_data(mode_config)