import math

# Velocities
OMNI_SPEED = 0.12
ROTATION_SPEED = 1
LU_SPEED = 0.12

DT = 0.1

scale = 1

# VSF parameters
L_VSS = scale * 77 * 10**(-3)  # VSS length
# L_VSS = math.pi/6
L_CONN = scale * 32 * 10**(-3)
D_BRIDGE = scale * 7 * 10**(-3)  # bridge width
L_VSF = 2 * L_VSS  # VSF length

# LU parameters
LU_SIDE = scale * 42 * 10**(-3)  # block side length
LU_R = LU_SIDE * math.sqrt(2) / 2 # Distance between LU center and its corner
LU_ALPHA = math.radians(-135) # Angle between LU orientation and r

# Wheels parameters
WHEEL_R = scale * 10 * 10**(-3)  # wheel radius
WHEEL_TH = scale * 12 * 10**(-3)  # wheel thickness
WHEEL_MARGIN = scale * 10**(-3)

BETA = [math.pi / 2, math.pi, -math.pi / 2, math.pi]

H1 = L_CONN + LU_SIDE - WHEEL_R
H2 = LU_SIDE + WHEEL_TH / 2 + WHEEL_MARGIN
H3 = L_CONN - WHEEL_TH / 2 - WHEEL_MARGIN
H4 = WHEEL_R

# Wheels coordinates w.r.t. to VSB end frames {b_j}
bj_Q_w = [[-H1, -H3, H3, H1],
          [-H2, -H4, -H4, -H2]]


# Constants of logarithmic spirals

# SPIRAL_COEF = [[2.3250 * L_VSS, 3.3041 * L_VSS,
#                 2.4471 * L_VSS], [0.3165, 0.083, 0.2229]]
SPIRAL_COEF = [[0.059, 0.1369, 0.1227], 
               [0.18, 0.1217, 0.2664]]

SPIRAL_PHI0 = [0.18, 0.1, 0.19]

SPIRAL_CENTRE = [-0.1223 * L_VSS, 0.1782 * L_VSS]

# M = [3 / 2, 1, 3 / 4]
M = [math.pi/2.0, math.pi/2.8, math.pi/2.14]

# Motive tracling data
M_POS = ['marker_x', 'marker_y', 'marker_z']
RB_POS = ['x', 'y', 'z']
RB_PARAMS = ['a', 'b', 'c', 'd']
RB_ANGLES = ['roll', 'pitch', 'yaw']

# Coords of the real LU center w.r.t. the rb position
HEAD_CENTER_R = scale * 0.01074968
HEAD_CENTER_ANGLE = math.radians(-60.2551187)

CARDIOID_A = [0.021, 0.049, 0.042]
CARDIOID_TH_MIN = [2.42, 2.19, 1.73]
CARDIOID_TH_MAX = [3.87, 4.09, 4.56]
CARDIOID_OFFSET = [0.006, 0.042, 0.015]