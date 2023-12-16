import numpy as np
from typing import List

def __bezierTwoPoints(t: (float, int), P1: np.ndarray, P2: np.ndarray) -> np.ndarray:
    Q1 = (1 - t) * P1 + t * P2
    return Q1

def __bezierPoints(t: (float, int), points: List[np.ndarray]) -> List[np.ndarray]:
    newpoints = []
    for i1 in range(0, len(points) - 1):
        newpoints += [__bezierTwoPoints(t, points[i1], points[i1 + 1])]
    return newpoints

def __bezierPoint(t: (float, int), points: List[np.ndarray]) -> np.ndarray:
    newpoints = points
    while len(newpoints) > 1:
        newpoints = __bezierPoints(t, newpoints)

    return newpoints[0]

def __bezierCurve(t_values: list, points: List[np.ndarray]) -> List[np.ndarray]:
    curve = np.array([[0.0] * len(points[0])])
    for t in t_values:
        curve = np.append(curve, [__bezierPoint(t, points)], axis=0)
    curve = np.delete(curve, 0, 0)
    
    return curve

def generateCurve(startPoint=[0, 0], lim=[[-2, 2], [-2, 2]]) -> tuple[np.ndarray, np.ndarray]:
    xlim = lim[0]
    ylim = lim[1]
    cp = [startPoint]
    ncp = 3 # Number of control points
    
    while len(cp) < ncp:
        x = np.random.rand() * (xlim[1] - xlim[0]) + xlim[0]
        y = np.random.rand() * (ylim[1] - ylim[0]) + ylim[0]
        cp.append([x, y])

    cp = np.array(cp)
    # cp = np.array([[0, 2], [2, 8], [6, 6], [4, 4], [2, 2], [6, 0], [8, 4], [10, 8], [8, 10], [6, 9]])
    t_points = np.arange(0, 1, 0.01)
    curve = __bezierCurve(t_points, cp)
    
    return curve

