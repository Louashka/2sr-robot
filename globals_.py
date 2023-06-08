import math
# Bridge parameters

L_VSS = 75.875 * 10**(-3)  # VSS length
L_VSS_CNN = 30 * 10**(-3)
D_BRIDGE = 7 * 10**(-3)  # bridge width
L_VSF = 2 * L_VSS  # VSF length

# Moving bloc parameters

BLOCK_SIDE = 42 * 10**(-3)  # block side length
WHEEL_R = 10 * 10**(-3)  # wheel radius
WHEEL_TH = 12 * 10**(-3)  # wheel thickness
WHEEL_MARGIN = 10**(-3)

BETA = [0, -math.pi / 2, math.pi / 2, 0]

H1 = L_VSS_CNN + BLOCK_SIDE - WHEEL_R
H2 = BLOCK_SIDE + WHEEL_TH / 2 + WHEEL_MARGIN
H3 = L_VSS_CNN - WHEEL_TH / 2 - WHEEL_MARGIN
H4 = WHEEL_R

# Wheels coordinates w.r.t. to VSB end frames {b_j}
bj_Q_w = [[-H1, -H3, H3, H1],
          [-H2, -H4, -H4, -H2]]


# Constants of logarithmic spirals

SPIRAL_COEF = [[2.3250 * L_VSS, 3.3041 * L_VSS,
                2.4471 * L_VSS], [0.3165, 0.083, 0.2229]]

SPIRAL_CENTRE = [-0.1223 * L_VSS, 0.1782 * L_VSS]

M = [3 / 2, 1, 3 / 4]
