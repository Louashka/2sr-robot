import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon as MatplotlibPolygon
from matplotlib.transforms import Affine2D
import matplotlib.cm as cm
from shapely import affinity
from shapely.geometry import Polygon as ShapelyPolygon

# Set figure parameters for high resolution
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['text.usetex'] = False
plt.rcParams['font.size'] = 24

agent_width = 0.07
agent_length = 0.34

def getRobotPolygon(pose):
    """Create robot polygon at given pose (x, y, theta)"""
    # Create rectangle centered at origin
    l, w = agent_length, agent_width
    points = [
        (-l/2, -w/2),
        (l/2, -w/2),
        (l/2, w/2),
        (-l/2, w/2),
        (-l/2, -w/2)  # Close the polygon by repeating first point
    ]
    # Create Shapely polygon
    robot = ShapelyPolygon(points)
    # Rotate and translate
    robot = affinity.rotate(robot, pose[2] * 180/np.pi)
    robot = affinity.translate(robot, pose[0], pose[1])
    return robot

def compute_potential(x, y, theta, obstacles, goal_pose, robot_length=2.5, robot_width=0.4):
    k_o, k_g = 0.15, 0.15
    epsilon = 1e-6
    
    # Obstacle potential
    U_obs = 0
    for obs in obstacles:
        obs_poly = ShapelyPolygon(obs)
        robot_poly = getRobotPolygon((x, y, theta))
        d = robot_poly.distance(obs_poly)
        center = np.mean(obs, axis=0)
        # d = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        # if d < d_thresh:
        alpha = np.arctan2(y - center[1], x - center[0]) - theta
        U_obs += abs(abs(alpha) - np.pi/2) / (d + epsilon)
    
    # Goal potential
    d_g = np.sqrt((x - goal_pose[0])**2 + (y - goal_pose[1])**2)
    U_goal = abs(theta - goal_pose[2]) / (d_g + epsilon)
    
    return k_o * U_obs + k_g * U_goal

def plot_robot(ax, x, y, theta, length=2.0, width=0.4, color='#FFD6B3', alpha=1.0, label=None):
    rect = Rectangle((-length/2, -width/2), length, width, facecolor=color, alpha=alpha, label=label)
    t = Affine2D().rotate(theta).translate(x, y)
    rect.set_transform(t + ax.transData)
    ax.add_patch(rect)


def point_in_polygon(x, y, polygon):
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

# Set up the environment
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(-3.9, 7.2)
ax.set_ylim(-3.9, 7.2)
# ax.set_xticks([])  # Remove x-axis ticks
# ax.set_yticks([])  # Remove y-axis ticks

# Define obstacles with 4 corners
obstacles = [
    np.array([[-0.6, -2.4], [1.2, -2], [1.3, -0.6], [-1.2, -1.8]]),  # First obstacle
    np.array([[-1.6, 1.0], [0.8, 1.7], [0.6, 2.3], [-1.6, 2.1]])  # Second obstacle
]

# Plot obstacles
for i, obs in enumerate(obstacles):
    if i == 0:
        ax.add_patch(MatplotlibPolygon(obs, facecolor='#CCCCCC', label='Obstacles'))
    else:
        ax.add_patch(MatplotlibPolygon(obs, facecolor='#CCCCCC'))

# Define goal pose
goal_pose = [5, 5, np.pi/3]
theta0 = np.pi/6

# Create vector field
x = np.linspace(-4, 7, 25)
y = np.linspace(-4, 7, 25)
X, Y = np.meshgrid(x, y)

# Compute optimal orientation at each point
U = np.zeros_like(X)
V = np.zeros_like(Y)
potentials = np.zeros_like(X)

for i in range(len(x)):
    for j in range(len(y)):
        # Check if point is inside any obstacle
        point_in_obs = False
        for obs in obstacles:
            if point_in_polygon(X[i,j], Y[i,j], obs):
                point_in_obs = True
                break
        
        if point_in_obs:
            continue
            
        min_potential = float('inf')
        best_theta = 0
        
        # Sample different orientations
        for theta in np.linspace(-np.pi, np.pi, 16):
            potential = compute_potential(X[i,j], Y[i,j], theta, obstacles, goal_pose)
            if potential < min_potential:
                min_potential = potential
                best_theta = theta
        
        U[i,j] = np.cos(best_theta)
        V[i,j] = np.sin(best_theta)
        potentials[i,j] = min_potential

# Normalize potentials for arrow scaling
potentials = (potentials - np.min(potentials)) / (np.max(potentials) - np.min(potentials))
scale = 0.2  # Adjust this value to change the maximum arrow length

# Second pass: smooth the vector field
U_smooth = np.copy(U)
V_smooth = np.copy(V)
kernel_size = 2  # Adjust this for more/less smoothing

for i in range(len(x)):
    for j in range(len(y)):
        if U[i,j] == 0 and V[i,j] == 0:  # Skip obstacle points
            continue
            
        # Get neighboring vectors
        i_start = max(0, i - kernel_size)
        i_end = min(len(x), i + kernel_size + 1)
        j_start = max(0, j - kernel_size)
        j_end = min(len(y), j + kernel_size + 1)
        
        # Average neighboring vectors
        u_sum = 0
        v_sum = 0
        count = 0
        
        for ni in range(i_start, i_end):
            for nj in range(j_start, j_end):
                if U[ni,nj] != 0 or V[ni,nj] != 0:  # Only consider non-obstacle points
                    u_sum += U[ni,nj]
                    v_sum += V[ni,nj]
                    count += 1
        
        if count > 0:
            # Normalize the averaged vector
            magnitude = np.sqrt((u_sum/count)**2 + (v_sum/count)**2)
            if magnitude > 0:
                U_smooth[i,j] = (u_sum/count) / magnitude
                V_smooth[i,j] = (v_sum/count) / magnitude

# Replace original vectors with smoothed vectors
U = U_smooth
V = V_smooth

# # Plot vector field with varying lengths
# for i in range(len(x)):
#     for j in range(len(y)):
#         if U[i,j] != 0 or V[i,j] != 0:  # Skip points inside obstacles
#             ax.arrow(X[i,j], Y[i,j], 
#                     U[i,j] * scale,
#                     V[i,j] * scale,
#                     head_width=0.06, head_length=0.06, 
#                     fc='grey', ec='grey', alpha=0.6)

plt.quiver(X,Y,U,V, color='#2A60C2')

# Example robot poses
robot_poses = [
    [0, 0, theta0],  # Robot between obstacles
    [goal_pose[0], goal_pose[1], goal_pose[2]]  # Robot at goal
]

# Plot robots with different alphas
plot_robot(ax, robot_poses[0][0], robot_poses[0][1], robot_poses[0][2], alpha=1, 
          label='Robot polygon')
plot_robot(ax, robot_poses[1][0], robot_poses[1][1], robot_poses[1][2], alpha=0.6,
          label='Target')

# Plot goal position and orientation
ax.plot(goal_pose[0], goal_pose[1], '*', color='#FF8C89', markersize=15)
ax.arrow(goal_pose[0], goal_pose[1], 
         1.0*np.cos(goal_pose[2]), 1.0*np.sin(goal_pose[2]),
         head_width=0.3, head_length=0.3, fc='#FF8C89', ec='#FF8C89')

plt.grid(True)
# plt.title('Artificial Potential Field Orientation Optimization')
# plt.xlabel('X')
# plt.ylabel('Y')
ax.set_aspect('equal')
plt.legend(loc='upper left')
# plt.show()
# Save as high-resolution PDF
plt.savefig('apf_orientation.pdf', format='pdf', dpi=150, bbox_inches='tight')
plt.close()