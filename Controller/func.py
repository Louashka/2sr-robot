import numpy as np

def normalizeAngle(angle: float) -> float:
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle

def close2Goal(current: list, target: list) -> bool:
    status = True

    # Calculate Euclidean distance between current and target (x, y)
    distance = np.linalg.norm(np.array(current) - np.array(target))
    
    # Define thresholds for position and orientation
    distance_threshold = 0.02

    if distance > distance_threshold:
        status = False
    
    print(f"Distance to goal: {distance:.3f} m\n")

    return status

def close2Shape(current_k: list, target_k: list) -> bool:
    status = True

    k1_diff = abs(current_k[0] - target_k[0])
    k2_diff = abs(current_k[1] - target_k[1])
    print(f'k1 diff: {k1_diff}')
    print(f'k2 diff: {k2_diff}\n')

    if k1_diff > 4.5 or k2_diff > 4.5:
        status = False
    return status