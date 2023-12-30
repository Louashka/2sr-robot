from Model.frame import Frame
from typing import List

class Marker():
    def __init__(self, marker_id: int, x: float, y: float) -> None:
        self.__marker_id = marker_id
        self.x = x
        self.y = y

    @property
    def marker_id(self) -> int:
        return self.__marker_id

    @property
    def x(self) -> float:
        return self.__x
    
    @x.setter
    def x(self, value: float) -> None:
        self.__x = value

    @property
    def y(self) -> float:
        return self.__y
    
    @y.setter
    def y(self, value: float) -> None:
        self.__y = value

    @property
    def position(self) -> list:
        return [self.x, self.y]

class Wheel(Frame):
    def __init__(self, agent_id: int, wheel_id: int, pose: list) -> None:
        self.__agent_id = agent_id
        self.__wheel_id = wheel_id
        super().__init__(pose)

    @property
    def agent_id(self) -> int:
        return self.__agent_id
    
    @property
    def wheel_id(self) -> int:
        return self.__wheel_id

class LU(Frame):
    def __init__(self, agent_id: int, pose: list, wheels: List[Wheel], marker_id=-1) -> None:
        self.__agent_id = agent_id
        super().__init__(pose)
        self.wheels = wheels
        self.__marker_id = marker_id

    def __str__(self) -> str:
        return '(' + ', '.join(map(str, self.pose)) + ')'

    @property
    def agent_id(self) -> int:
        return self.__agent_id
    
    @property
    def wheels(self) -> List[Wheel]:
        return self.__wheels
    
    @wheels.setter
    def wheels(self, value: List[Wheel]) -> None:
        for wheel in value:
            if not isinstance(wheel, Wheel):
                raise ValueError('Wrong type of wheels!')
        
        self.__wheels = value

    @property
    def marker_id(self) -> int:
        return self.__marker_id

class VSF:
    def __init__(self, agent_id: int, markers: List[Marker]) -> None:
        self.__agent_id = agent_id
        self.markers = markers

    def __str__(self) -> str:
        return '[' + ', '.join(map(str, self.k)) + ']'

    @property
    def agent_id(self) -> int:
        return self.__agent_id
    
    @property
    def markers(self) -> List[Marker]:
        return self.__markers
    
    @markers.setter
    def markers(self, value: List[Marker]) -> None:
        for val in value:
            if not isinstance(val, Marker):
                raise ValueError('Wrong type of markers!')
        
        self.__markers = value
    
    @property
    def k(self) -> List[float]:
        return [0.0, 0.0]
    
    @property
    def markers_line(self) -> list:
        x = []
        y = []
        for marker in self.markers:
            x.append(marker.x)
            y.append(marker.y)

        return [x, y]

class Robot(Frame):
    def __init__(self, id, pose: list, head: LU, tail: LU, vsf: VSF, status=True) -> None:
        self.__id = id # Correspond to the model_id
        super().__init__(pose)
        self.__head = head
        self.__tail = tail
        self.__vsf = vsf
        self.status = status

    def __str__(self) -> str:
        str_keys = ['id: ', ', pose: (', '), head: ', ', tail: ', ', VSF: ']
        str_values = [str(self.id), ', '.join(map(str, self.pose)), str(self.head), str(self.tail), str(self.vsf)]

        result = ''
        for str_key, str_value in zip(str_keys, str_values):
            result += str_key + str_value

        return result

    @property
    def id(self) -> int:
        return self.__id
    
    @property
    def head(self) -> LU:
        return self.__head
    
    @property
    def tail(self) -> LU:
        return self.__tail
    
    @property
    def vsf(self) -> VSF:
        return self.__vsf
    
    @property
    def status(self) -> bool:
        return self.__status
    
    @status.setter
    def status(self, value: bool) -> None:
        self.__status = value
    
    @property
    def config(self) -> list:
        return self.pose + self.vsf.k
    
    @property
    def allWheels(self) -> List[Wheel]:
        return self.head.wheels + self.tail.wheels
           