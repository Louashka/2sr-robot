# =============================================================================
#
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
#       - 2D trajectory plots visualizing the reachable (position) workspace of the robot.
#       - A 3D scatter plot visualizing a constrained manifold in 4D configuration space.
# USAGE:
#   1.  Set up the desried robot stiffness (robot.stiff1 and robot.stiff2).
#   2.  Determine the motion mode that corresponds to the chosen stiffness.
#   3.  To generate new data, uncomment the `collect_data(robot, mode_config)` line.
#   4.  To analyze data for the chosen motion mode, run the script with the 
#       `analyse_data(mode_config)` line active. To analyse the whole data accross all 
#       modes run 'analyse_all_data()'.
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
import matplotlib.colors as mcolors
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from entities import global_var as gv, robot_state
import kinematics

# =============================================================================
# SCRIPT CONFIGURATION
# =============================================================================

FONT_SIZE_CONFIG = {
    "axes.titlesize": 28,      # Fontsize of the axes title
    "axes.labelsize": 26,      # Fontsize of the x and y labels
    "xtick.labelsize": 26,     # Fontsize of the x-tick labels
    "ytick.labelsize": 26,     # Fontsize of the y-tick labels
    "legend.fontsize": 18,     # Fontsize of the legend
    "figure.titlesize": 28     # Fontsize of the main figure title
}
plt.rcParams.update(FONT_SIZE_CONFIG)

VARS_COLORS = {
    "x": "viridis",
    "y": "viridis",
    "theta": "plasma",
    "k": "coolwarm",
}

# This dictionary defines the parameters for each flexible operational mode of the robot.
MODES = {
    (1, 0): {
        'file_path': "analysis/data/semi_flex_1_data.parquet",
        'hm_fig_path': "analysis/figures/semi_flex_1_heatmap.pdf",
        'workspace_fig_path': "analysis/figures/semi_flex_1_workspace.pdf",
        'manifold_fig_path': "analysis/figures/semi_flex_1_manifold.png",
        'state_vars': ['x', 'y', 'theta', 'k1'],
        'curvature_to_display': "k1",
        'k_label': r"$\kappa_1$",
        'title': 'Flex Mode 1'
    },
    (0, 1): {
        'file_path': "analysis/data/semi_flex_2_data.parquet",
        'hm_fig_path': "analysis/figures/semi_flex_2_heatmap.pdf",
        'workspace_fig_path': "analysis/figures/semi_flex_2_workspace.pdf",
        'manifold_fig_path': "analysis/figures/semi_flex_2_manifold.png",
        'state_vars': ['x', 'y', 'theta', 'k2'],
        'curvature_to_display': "k2",
        'k_label': r"$\kappa_2$",
        'title': 'Flex Mode 2'
    },
    (1, 1): {
        'file_path': "analysis/data/flex_data.parquet",
        'hm_fig_path': "analysis/figures/flex_heatmap.pdf",
        'workspace_fig_path': "analysis/figures/flex_workspace.pdf",
        'manifold_fig_path': "analysis/figures/flex_manifold.png",
        'state_vars': ['x', 'y', 'theta', 'k1'],
        'curvature_to_display': "k1",
        'k_label': r"$\kappa_1$",
        'title': 'Flex Mode 3'
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
                command_vector = [0.0, 0.0, 0.0, v1, v2]
                J = kinematics_handler.get_unified_jacobian(robot, robot.stiffness)
                q_dot = J.dot(command_vector)
                new_config = robot.config + q_dot * dt
                robot.config = new_config.tolist()

                # --- Data Storage ---
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
    print(f"\nLoading data from {file_path}...")
    if not os.path.exists(file_path):
        print(f"\nError: Data file not found at {file_path}.")
        print("Please run the `collect_data` function first.")
        return

    df = pd.read_parquet(file_path)
    var_scale = 100 # Scaling factor for visualization (m to cm)

    # --- Generate and Display Plots ---
    hm_fig = plot_heatmaps(df, config, var_scale)
    workspace_fig, workspace_ax  = plot_workspace(df, config, var_scale)
    manifold_fig = plot_manifold(df, config, var_scale)

    print("\nSaving figures...")
    # hm_fig.savefig(config["hm_fig_path"], dpi=150, transparent=True)
    # workspace_fig.savefig(config["workspace_fig_path"], dpi=150, transparent=True, bbox_inches='tight')
    # manifold_fig.savefig(config["manifold_fig_path"], dpi=300, transparent=True, bbox_inches='tight')

    print("\nDisplaying generated figures...")
    plt.show()

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

    fig, axes = plt.subplots(2, 2, figsize=(18, 13))
    # fig.suptitle(f'Control Space Performance Metrics ({config["title"]})')

    v1_min, v1_max = df['v1'].min() * var_scale, df['v1'].max() * var_scale
    v2_min, v2_max = df['v2'].min() * var_scale, df['v2'].max() * var_scale
    plot_extent = [v1_min, v1_max, v2_min, v2_max]
    
    # TOP ROW: DISPLACEMENT (Shared 'viridis' colormap)
    vmax_disp = max(pivot_x.max().max(), pivot_y.max().max())
    
    ax = axes[0, 0]
    im = ax.imshow(pivot_x.values, extent=plot_extent, cmap=VARS_COLORS['x'], aspect='equal', origin='lower', vmin=0, vmax=vmax_disp)
    # ax.set_title('Absolute Max Displacement in X')
    ax.set_ylabel(r'$v_2$ [cm/s]')
    ax.set_yticks([-7.0, 0.0, 7.0])
    ax1_cbar = fig.colorbar(im, ax=ax)
    ax1_cbar.set_label('Max |x| [cm]', labelpad=35)

    ax = axes[0, 1]
    im = ax.imshow(pivot_y.values, extent=plot_extent, cmap=VARS_COLORS['y'], aspect='equal', origin='lower', vmin=0, vmax=vmax_disp)
    # ax.set_title('Absolute Max Displacement in Y')
    ax2_cbar = fig.colorbar(im, ax=ax)
    ax2_cbar.set_label('Max |y| [cm]', labelpad=35)

    # BOTTOM ROW: ROTATION & CURVATURE (Independent colormaps)
    ax = axes[1, 0]
    im = ax.imshow(pivot_rot.values, extent=plot_extent, cmap=VARS_COLORS['theta'], aspect='equal', origin='lower')
    # ax.set_title('Total Rotation')
    ax.set_ylabel(r'$v_2$ [cm/s]')
    ax.set_yticks([-7.0, 0.0, 7.0])
    ax3_cbar = fig.colorbar(im, ax=ax)
    ax3_cbar.set_label(r'Total $\theta$ [rad]', labelpad=37)

    ax = axes[1, 1]
    im = ax.imshow(pivot_curv.values, extent=plot_extent, cmap=VARS_COLORS['k'], aspect='equal', origin='lower')
    # ax.set_title(f'Final Curvature ({curv_to_display})')
    ax.set_xlabel(r'$v_1$ [cm/s]')
    fig.colorbar(im, ax=ax, label=f'Final {config['k_label']} ($m^{{-1}}$)')

    axes[0, 0].set_xticklabels([])
    axes[0, 1].set_xticklabels([])
    axes[0, 1].set_yticklabels([])
    axes[1, 1].set_yticklabels([])

    plt.tight_layout()

    return fig

def plot_workspace(df: pd.DataFrame, config: dict, var_scale: float):
    """
    Generates a plot of the full reachable workspace.

    Args:
        df: DataFrame containing the trajectory data.
        config: Configuration dictionary (used for title, etc.).
        var_scale: Scaling factor for position variables (e.g., 100 to convert m to cm).

    Returns:
        A tuple containing the matplotlib Figure and Axes objects (fig, ax).
    """
    print("Generating workspace plot...")
    
    # Using plt.subplots is a convenient way to create a figure and axes at once
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.canvas.manager.set_window_title('Reachable Workspace')
    
    # ax.set_title(f'Reachable Workspace ({config.get("title", "")})', pad=15)
    ax.set_xlabel(f'x [cm]')
    ax.set_ylabel(f'y [cm]')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_aspect('equal', adjustable='box')

    # Subsample and group data for plotting
    sub_df_grouped = df.iloc[::20].groupby(["v1", "v2"])

    # Plot every trajectory with a light, transparent color
    for _, traj in sub_df_grouped:
        ax.plot(
            traj['x'] * var_scale,
            traj['y'] * var_scale,
            color='#4169E1', # Royal blue
            alpha=0.05,
            linewidth=1.0
        )
    
    plt.tight_layout()

    return fig, ax

def plot_manifold(df: pd.DataFrame, config: dict, var_scale: float, ax_workspace=None):
    """
    Generates a 3D plot of the configuration manifold.

    Args:
        df: DataFrame containing the trajectory data.
        config: Configuration dictionary.
        var_scale: Scaling factor for position variables.
        ax_workspace: (Optional) The Axes object from a 2D workspace plot.
                      If provided, the x and y limits of the 3D plot will be
                      synchronized to match.

    Returns:
        The matplotlib Figure object for the 3D plot.
    """
    print("Generating 3D manifold plot...")

    curv_to_display = config.get("curvature_to_display", "k1")
    k_label = config["k_label"]

    # --- Plotting Setup ---
    fig = plt.figure(figsize=(9, 8))
    fig.canvas.manager.set_window_title('Configuration Manifold')
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    
    # --- Data Preparation ---
    sub_df_grouped = df.groupby(["v1", "v2"])
    sub_df_3d = sub_df_grouped.apply(lambda x: x.iloc[::10], include_groups=False).reset_index(drop=True)
    
    # --- Scatter Plot ---
    ax.scatter(
        sub_df_3d['x'] * var_scale, sub_df_3d['y'] * var_scale, sub_df_3d['theta'],
        c=sub_df_3d[curv_to_display],
        cmap='plasma',
        s=0.2,
        alpha=0.2
    )

    # --- Colorbar ---
    norm = mcolors.Normalize(vmin=sub_df_3d[curv_to_display].min(), vmax=sub_df_3d[curv_to_display].max())
    sm = plt.cm.ScalarMappable(cmap='plasma', norm=norm)
    fig.colorbar(sm, ax=ax, shrink=0.6, label=k_label, fraction=0.03, aspect=30, pad=0.12)
    
    # --- Labels and Title ---
    # ax.set_title(f'Constrained Configuration Manifold ({config.get("title", "")})')
    ax.set_xlabel('x [cm]', labelpad=20)
    ax.set_ylabel('y [cm]', labelpad=20)
    ax.set_zlabel('$\\theta$ [rad]', labelpad=15)

    ax.set_yticks([-6.0, 0.0, 6.0])
    ax.set_zticks([-5.0, 0.0, 5.0])

    # --- Synchronize Axes (if workspace axes are provided) ---
    if ax_workspace:
        print("Synchronizing manifold axes with workspace axes.")
        ax.set_xlim(ax_workspace.get_xlim())
        ax.set_ylim(ax_workspace.get_ylim())

    plt.tight_layout()

    return fig

def analyse_all_data():
    """
    Generates and saves the combined heatmap figure for all modes.
    """
    var_scale = 100  # Scaling factor for visualization (m to cm)

    # Generate the combined figure
    combined_fig = plot_all_modes_heatmaps(MODES, var_scale)

    # Save the combined figure
    save_path = "analysis/figures/control_space_all_modes.pdf"
    print(f"\nSaving combined figure to {save_path}...")
    if not os.path.exists("figures"):
        os.makedirs("figures")
    combined_fig.savefig(save_path, dpi=150, transparent=True, bbox_inches='tight')

    print("\nDisplaying generated figure...")
    plt.show()

def plot_all_modes_heatmaps(modes_config: dict, var_scale: float):
    """
    Generates a 4x3 grid of heatmaps for all operational modes.

    Each row corresponds to a performance metric, and each column corresponds
    to a robot mode, allowing for direct visual comparison. Color scales are
    normalized across modes for each metric.

    Args:
        modes_config (dict): The main MODES dictionary containing configs for all modes.
        var_scale (float): Scaling factor for visualization (e.g., 100 for m to cm).
    
    Returns:
        matplotlib.figure.Figure: The generated figure object.
    """
    print("Generating combined heatmap figure for all modes...")
    num_modes = len(modes_config)
    
    # --- Part 1: Pre-compute global color limits for consistent scaling ---
    global_max_disp = 0
    global_max_rot = 0
    global_min_curv, global_max_curv = float('inf'), float('-inf')

    for mode, config in modes_config.items():
        file_path = config['file_path']
        if not os.path.exists(file_path):
            print(f"Warning: Data file not found for mode {mode} at {file_path}. Skipping.")
            continue
        
        df = pd.read_parquet(file_path)
        summary_df = df.groupby(["v1", "v2"]).agg(
            max_abs_x=('x', lambda s: s.abs().max()),
            max_abs_y=('y', lambda s: s.abs().max()),
            total_rotation=('theta', lambda t: abs(t.iloc[-1] - t.iloc[0])),
            final_curvature=(config.get("curvature_to_display", "k1"), 'last')
        ).reset_index()

        current_max_disp = max(summary_df['max_abs_x'].max(), summary_df['max_abs_y'].max()) * var_scale
        global_max_disp = max(global_max_disp, current_max_disp)
        
        global_max_rot = max(global_max_rot, summary_df['total_rotation'].max())
        
        global_min_curv = min(global_min_curv, summary_df['final_curvature'].min())
        global_max_curv = max(global_max_curv, summary_df['final_curvature'].max())

    symmetric_curv_limit = max(abs(global_min_curv), abs(global_max_curv))

    # --- Part 2: Create the figure and plot data for each mode ---
    fig, axes = plt.subplots(4, num_modes, figsize=(5 * num_modes, 18), sharex=True)

    mode_list = list(modes_config.items())

    for col_idx, (mode, config) in enumerate(mode_list):
        print(f"  Plotting Mode: {config['title']}")
        file_path = config['file_path']
        if not os.path.exists(file_path):
            continue

        df = pd.read_parquet(file_path)
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

        v1_min, v1_max = df['v1'].min() * var_scale, df['v1'].max() * var_scale
        v2_min, v2_max = df['v2'].min() * var_scale, df['v2'].max() * var_scale
        plot_extent = [v1_min, v1_max, v2_min, v2_max]
        
        # --- Plotting on the grid ---
        # Set column title on the top-most plot
        axes[0, col_idx].set_title(config['title'], fontweight='bold', pad=15)

        # Row 0: Max |x|
        im0 = axes[0, col_idx].imshow(pivot_x.values, extent=plot_extent, cmap='viridis', aspect='equal', origin='lower', vmin=0, vmax=global_max_disp)
        
        # Row 1: Max |y|
        im1 = axes[1, col_idx].imshow(pivot_y.values, extent=plot_extent, cmap='viridis', aspect='equal', origin='lower', vmin=0, vmax=global_max_disp)
        
        # Row 2: Total Rotation
        im2 = axes[2, col_idx].imshow(pivot_rot.values, extent=plot_extent, cmap='plasma', aspect='equal', origin='lower', vmin=0, vmax=global_max_rot)
        
        # Row 3: Final Curvature
        im3 = axes[3, col_idx].imshow(pivot_curv.values, extent=plot_extent, cmap='coolwarm', aspect='equal', origin='lower', vmin=-symmetric_curv_limit, vmax=symmetric_curv_limit)

        # --- Clean up labels ---
        # Set x-axis label only on the bottom row
        axes[3, col_idx].set_xlabel(r'$v_1$ [cm/s]')
        axes[3, col_idx].set_xticks([-7.0, 0.0, 7.0])
        
        # Set y-axis labels and ticks only on the first column
        if col_idx == 0:
            axes[0, 0].set_ylabel(r'$v_2$ [cm/s]')
            axes[1, 0].set_ylabel(r'$v_2$ [cm/s]')
            axes[2, 0].set_ylabel(r'$v_2$ [cm/s]')
            axes[3, 0].set_ylabel(r'$v_2$ [cm/s]')
            for row_idx in range(4):
                axes[row_idx, 0].set_yticks([-7.0, 0.0, 7.0])
        else:
            for row_idx in range(4):
                axes[row_idx, col_idx].set_yticklabels([])

    # --- Part 3: Add shared colorbars for each row ---
    fig.colorbar(im0, ax=axes[0, 2], fraction=0.03, aspect=30, pad=0.03).set_label('Max |x| [cm]', labelpad=37)
    fig.colorbar(im1, ax=axes[1, 2], fraction=0.03, aspect=30, pad=0.03).set_label('Max |y| [cm]', labelpad=37)
    fig.colorbar(im2, ax=axes[2, 2], fraction=0.03, aspect=30, pad=0.03).set_label(r'Total $\theta$ [rad]', labelpad=37)
    fig.colorbar(im3, ax=axes[3, 2], fraction=0.03, aspect=30, pad=0.03).set_label(f'Final {config['k_label']} ($m^{{-1}}$)')
    
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

    # To analyze the current mode, run the line below.
    analyse_data(mode_config)

    # ... or analyze data accross all modes.
    # analyse_all_data()