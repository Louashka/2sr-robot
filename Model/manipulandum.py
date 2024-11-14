from Model.frame import Frame
import numpy as np
import math
import pandas as pd
from typing import List
import cv2 as cv
from pyefd import elliptic_fourier_descriptors as efd

dir = 'Experiments/Data/Contours/'
shapes = {11: {'type': 'cheescake', 'path': dir + 'cheescake_contour.csv'},
          12: {'type': 'ellipse', 'path': dir + 'ellipse_contour.csv'},
          13: {'type': 'heart', 'path': dir + 'heart_contour.csv'},
          14: {'type': 'bean', 'path': dir + 'bean_contour.csv'}}


class Shape(Frame):
    def __init__(self, id: int, pose: List[float]) -> None:
        super().__init__(pose[0], pose[1], pose[2])

        self.__id = id
        self.delta_theta = 0 
        self.m = 10

        self.__retrieveContour(shapes[id]['path'])        

    def __str__(self) -> str:
        response = 'id: ' + str(self.id) + ', pose: (' + ', '.join(map(str, self.pose)) + ')' 
        return response

    @property
    def id(self) -> int:
        return self.__id
    
    @property
    def heading_angle(self) -> float:
        return self.theta + self.delta_theta
    
    @property
    def pose_heading(self) -> List[float]:
        return self.position + [self.heading_angle]

    @property
    def rotation_matrix(self) -> np.ndarray:
        return np.array([[np.cos(self.theta), -np.sin(self.theta), 0],
                         [np.sin(self.theta), np.cos(self.theta), 0],
                         [0, 0, 1]])
    
    def __retrieveContour(self, path):
        contour_df = pd.read_csv(path)
        contour_r = contour_df['radius'].tolist()
        contour_theta = contour_df['phase_angle'].tolist()

        contour_params = [contour_r, contour_theta]
        
        points = []
        for r, phi in zip(contour_params[0], contour_params[1]):
            x = r * np.cos(phi)
            y = r * np.sin(phi)

            points.append([x, y])

        self.default_contour = np.array(points).T 
        self.coeffs = efd(self.default_contour.T, order = self.m)
    
    def geomCentre(self, points) -> list:
        ctr = np.array(points).reshape((-1,1,2))
        ctr = (10000.0 * ctr).astype(np.int32)

        M = cv.moments(ctr)
        cX = int(M["m10"] / M["m00"]) / 10000.0
        cY = int(M["m01"] / M["m00"]) / 10000.0

        r = math.hypot(cX, cY)
        phi = math.atan2(cY, cX)

        x = r * np.cos(phi)
        y = r * np.sin(phi)

        return [x, y]
    
    def __calcPerimeter(self, points) -> float:
        ctr = np.array(points).reshape((-1,1,2))
        ctr = (10000.0 * ctr).astype(np.int32)

        return cv.arcLength(ctr,True) / 10000.0

    @property
    def contour(self) -> np.ndarray:
        R = np.array([[np.cos(self.theta), -np.sin(self.theta)], 
                      [np.sin(self.theta), np.cos(self.theta)]])
        
        ctr = R.dot(self.default_contour) + np.array([self.position]).T

        return ctr
    
    @property
    def parametric_contour(self) -> tuple[np.ndarray, np.ndarray]:
        ctr = []
        s_array = np.linspace(0, 1, 200)
        for s in s_array:
            pos_target = self.getPoint(s)
            ctr.append(pos_target)

        ctr = np.array(ctr).T

        return s_array, ctr
    
    def getPoint(self, s: float) -> List[float]:
        coords = []

        for h in range(self.m):
            arg = 2 * np.pi * (h + 1) * s
            exp = np.array([[np.cos(arg)], [np.sin(arg)]])

            coef = self.coeffs[h,:].reshape(2, 2)
            coord_h = np.matmul(coef, exp).T

            coords.append(coord_h)

        point_normalised = sum(coords)[0]

        R = np.array([[np.cos(self.theta), -np.sin(self.theta)], 
                      [np.sin(self.theta), np.cos(self.theta)]])
        point = R.dot(np.array([point_normalised]).T) + np.array([self.position]).T
        point = point.T

        return point[0].tolist()
    
    def getTangent(self, s: float) -> float:
        dx = 0
        dy = 0

        for h in range(self.m):
            c = 2 * (h + 1) * np.pi
            arg = c * s
            exp = [-c * np.sin(arg),  c * np.cos(arg)]

            coef = self.coeffs[h,:]
            dx += coef[0] * exp[0] + coef[1] * exp[1]
            dy += coef[2] * exp[0] + coef[3] * exp[1]

        theta = np.arctan(dy/dx) + self.theta
        # Ensure the tangent points in the positive direction of traversing the contour
        # Calculate the vector perpendicular to the tangent
        # perp_vector = np.array([np.cos(theta + np.pi/2), np.sin(theta + np.pi/2)])
        theta_vector = np.array([np.cos(theta), np.sin(theta)])
        
        # Get a point slightly ahead on the contour
        s_ahead = (s + 0.01) % 1  # Ensure we wrap around if s is close to 1
        point_ahead = np.array(self.getPoint(s_ahead))
        
        # Calculate vector from current point to point ahead
        current_point = np.array(self.getPoint(s))
        direction_vector = point_ahead - current_point
        
        # Check if perpendicular vector points outwards
        if np.dot(theta_vector, direction_vector) < 0:
            theta += np.pi  # Add 180 degrees if pointing outwards

        
        # Normalize theta to be between 0 and 2π
        theta = theta % (2 * np.pi)
        return theta

    

    
