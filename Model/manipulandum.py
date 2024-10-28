import numpy as np
import math
from Model.frame import Frame
from typing import List
import cv2 as cv
from pyefd import elliptic_fourier_descriptors as efd

class Shape(Frame):
    def __init__(self, id: int, pose: List[float], contour_params=[]) -> None:
        self.__id = id
        super().__init__(pose[0], pose[1], pose[2])
        self.m = 10
        self.contour_params = contour_params
        self.default_contour, self.perimeter = self.__paramsToCoords()
        
        self.coeffs = efd(self.default_contour.T, order = self.m)

        self.lin_vel_x = 0
        self.lin_vel_y = 0
        self.ang_vel = 0      

    def __str__(self) -> str:
        response = 'id: ' + str(self.id) + ', pose: (' + ', '.join(map(str, self.pose)) + ')' 
        return response

    @property
    def id(self) -> int:
        return self.__id
    
    @property
    def contour_params(self) -> list:
        return self.__contour_params
    
    @contour_params.setter
    def contour_params(self, value) -> None:
        self.__contour_params = value

    @property
    def velocity(self) -> list:
        return [self.lin_vel_x, self.lin_vel_y, self.ang_vel]

    @property
    def rotation_matrix(self) -> np.ndarray:
        return np.array([[np.cos(self.theta), -np.sin(self.theta), 0],
                         [np.sin(self.theta), np.cos(self.theta), 0],
                         [0, 0, 1]])

    def __paramsToCoords(self) -> tuple[np.ndarray, float]:
        points = []
        for r, phi in zip(self.contour_params[0], self.contour_params[1]):
            x = r * np.cos(phi)
            y = r * np.sin(phi)

            points.append([x, y])

        # geom_centre = self.geomCentre(points)
        default_contour = np.array(points).T 

        perimeter = self.__calcPerimeter(points)

        return default_contour, perimeter
    
    def geomCentre(self, points) -> list:
        ctr = np.array(points).reshape((-1,1,2))
        ctr = (10000.0 * ctr).astype(np.int32)

        M = cv.moments(ctr)
        cX = int(M["m10"] / M["m00"]) / 10000.0
        cY = int(M["m01"] / M["m00"]) / 10000.0

        self.r = math.hypot(cX, cY)
        self.phi = math.atan2(cY, cX)

        x = self.r * np.cos(self.phi)
        y = self.r * np.sin(self.phi)

        # self.x += r * np.cos(self.theta + phi)
        # self.y += r * np.sin(self.theta + phi)

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
        s_array = np.linspace(0, 1)
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
        perp_vector = np.array([np.cos(theta + np.pi/2), np.sin(theta + np.pi/2)])
        
        # Get a point slightly ahead on the contour
        s_ahead = (s + 0.01) % 1  # Ensure we wrap around if s is close to 1
        point_ahead = np.array(self.getPoint(s_ahead))
        
        # Calculate vector from current point to point ahead
        current_point = np.array(self.getPoint(s))
        direction_vector = point_ahead - current_point
        
        # Check if perpendicular vector points outwards
        if np.dot(perp_vector, direction_vector) < 0:
            theta += np.pi  # Add 180 degrees if pointing outwards

        
        # Normalize theta to be between 0 and 2π
        theta = theta % (2 * np.pi)
        return theta
    
    # def update(self, q_dot: list, dt: float):

    #     self.x += q_dot[0] * dt
    #     self.y += q_dot[1] * dt
    #     self.theta += q_dot[2] * dt

    def update(self, acc: list):
        dt = 0.05

        q_dot = self.rotation_matrix.dot(np.array(self.velocity).T)

        self.x += q_dot[0] * dt
        self.y += q_dot[1] * dt
        self.theta += q_dot[2] * dt

        self.lin_vel_x += acc[0] * dt
        self.lin_vel_y += acc[1] * dt
        self.ang_vel += acc[2] * dt
    

    
