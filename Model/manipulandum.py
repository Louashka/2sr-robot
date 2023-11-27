import numpy as np
from Model.frame import Frame

class Shape(Frame):
    def __init__(self, id, pose, contour_params=[]) -> None:
        self.__id = id
        super().__init__(pose)
        self.contour_params = contour_params

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
    def contour(self) -> list:
        contour_x, contour_y = [], []
        for r, theta in zip(self.contour_params[0], self.contour_params[1]):
            contour_x.append(self.x + r * np.cos(self.theta + theta))
            contour_y.append(self.y + r * np.sin(self.theta + theta))

        return [contour_x, contour_y]