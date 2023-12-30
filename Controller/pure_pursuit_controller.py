import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/lytaura/Documents/PolyU/Research/2SR/Version 1/Multi agent/Control/2sr-swarm-control')
from Model import manipulandum as mp
import path
import numpy as np

GOAL_RADIUS = 0.05

class GraspModel:
    def __init__(self) -> None:
        pass



if __name__ == "__main__":
    # Define the manipulandum shape
    heart_df = pd.read_csv('./Data/heart_contour.csv')[['distance', 'phase']].dropna()
    heart_r = heart_df['distance'].tolist()
    heart_theta = heart_df['phase'].tolist()
    heart = mp.Shape(11, [0.4, 0.32, 0.8 * np.pi], [heart_r, heart_theta])

    # Generate a random trajectory 
    trajectory = path.Trajectory(heart.position)

    # Determine contact points
    for s in np.linspace(0, 1):
        pos_target = heart.getPoint(s)
        plt.plot(pos_target[0], pos_target[1], '*k')
    # Calculate the contact point coordinates 
    # x_c, y_c = f(s) 
    q_target = []
    

    # Execute grasping by the robot
    

    plt.plot(heart.contour[0], heart.contour[1], '.r')
    plt.plot(heart.x, heart.y, 'or')
    plt.plot(trajectory.traj_x, trajectory.traj_y, '--k')
    plt.axis('equal')
    plt.show()

