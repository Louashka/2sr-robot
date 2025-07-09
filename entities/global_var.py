import math

# Velocities
OMNI_SPEED = 0.12
ROTATION_SPEED = 1
LU_SPEED = 0.12

DT = 0.1

# VSB parameters
L_VSS = 77 * 10**(-3)  # VSS length
L_CONN = 32 * 10**(-3)
D_BRIDGE = 7 * 10**(-3)  # bridge width
L_VSB = 2 * L_VSS  # VSF length

# LU parameters
LU_SIDE = 42 * 10**(-3)  # block side length
LU_R = LU_SIDE * math.sqrt(2) / 2 # Distance between LU center and its corner
LU_ALPHA = math.radians(-135) # Angle between LU orientation and r

# Wheels parameters
WHEEL_R = 10 * 10**(-3)  # wheel radius
WHEEL_TH = 12 * 10**(-3)  # wheel thickness
WHEEL_MARGIN = 10**(-3)

BETA = [math.pi / 2, math.pi, -math.pi / 2, math.pi]

H1 = L_CONN + LU_SIDE - WHEEL_R
H2 = LU_SIDE + WHEEL_TH / 2 + WHEEL_MARGIN
H3 = L_CONN - WHEEL_TH / 2 - WHEEL_MARGIN
H4 = WHEEL_R

# Wheels coordinates w.r.t. to VSB end frames {b_j}
bj_Q_w = [[-H1, -H3, H3, H1],
          [-H2, -H4, -H4, -H2]]

# Coords of the real LU center w.r.t. the rb position
HEAD_CENTER_R = 0.01074968
HEAD_CENTER_ANGLE = math.radians(-60.2551187)

CARDIOID_A = [0.021, 0.049, 0.042]
CARDIOID_TH_MIN = [2.42, 2.19, 1.73]
CARDIOID_TH_MAX = [3.87, 4.09, 4.56]
CARDIOID_OFFSET = [0.006, 0.042, 0.015]

# Motive tracking data
M_POS = ['marker_x', 'marker_y', 'marker_z']
RB_POS = ['x', 'y', 'z']
RB_PARAMS = ['a', 'b', 'c', 'd']
RB_ANGLES = ['roll', 'pitch', 'yaw']