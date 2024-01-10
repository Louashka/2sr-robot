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

    def __paramsToCoords(self) -> tuple[np.ndarray, float]:
        points = []
        for r, phi in zip(self.contour_params[0], self.contour_params[1]):
            x = r * np.cos(phi)
            y = r * np.sin(phi)

            points.append([x, y])

        geom_centre = self.__geomCentre(points)
        default_contour = np.array(points).T - np.array([geom_centre]).T

        perimeter = self.__calcPerimeter(points)

        return default_contour, perimeter
    
    def __geomCentre(self, points) -> list:
        ctr = np.array(points).reshape((-1,1,2))
        ctr = (10000.0 * ctr).astype(np.int32)

        M = cv.moments(ctr)
        cX = int(M["m10"] / M["m00"]) / 10000.0
        cY = int(M["m01"] / M["m00"]) / 10000.0

        r = math.hypot(cX, cY)
        phi = math.atan2(cY, cX)

        x = r * np.cos(phi)
        y = r * np.sin(phi)

        self.x += r * np.cos(self.theta + phi)
        self.y += r * np.sin(self.theta + phi)

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
    

    
