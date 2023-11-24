class Frame:
    def __init__(self, pose) -> None:
        self.x = pose[0]
        self.y = pose[1]
        self.theta = pose[2]
    
    @property
    def x(self) -> float:
        return self.__x
    
    @x.setter
    def x(self, value) -> None:
        self.__x = value

    @property
    def y(self) -> float:
        return self.__y
    
    @y.setter
    def y(self, value) -> None:
        self.__y = value

    @property
    def theta(self) -> float:
        return self.__theta
    
    @theta.setter
    def theta(self, value) -> None:
        self.__theta = value

    @property
    def position(self) -> list:
        return [self.x, self.y]
    
    @position.setter
    def position(self, value) -> None:
        if len(value) != 2:
            raise ValueError("Wrong number of position coordinates!")
        self.__x, self.__y = value

    @property
    def pose(self) -> list:
        return [self.x, self.y, self.theta]
    
    @pose.setter
    def pose(self, value) -> None:
        if len(value) != 3:
            raise ValueError("Wrong number of pose coordinates!")
        self.__x, self.__y, self.__theta = value

class Wheel(Frame):
    def __init__(self, agent_id, wheel_id, pose) -> None:
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
    def __init__(self, agent_id, pose, wheels) -> None:
        self.__agent_id = agent_id
        super().__init__(pose)
        self.__wheels = wheels

    def __str__(self) -> str:
        return '(' + ', '.join(map(str, self.pose)) + ')'

    @property
    def agent_id(self) -> int:
        return self.__agent_id
    
    @property
    def wheels(self) -> list:
        return self.__wheels

class VSF:
    def __init__(self, agent_id, k) -> None:
        self.__agent_id = agent_id
        self.k = k

    def __str__(self) -> str:
        return '[' + ', '.join(map(str, self.k)) + ']'

    @property
    def agent_id(self) -> int:
        return self.__agent_id
    
    @property
    def k(self) -> list:
        return self.__k
    
    @k.setter
    def k(self, value) -> None:
        if len(value) != 2:
            raise ValueError("Wrong number of VSF segments!")
        
        if not all(isinstance(val, float) for val in value):
            raise ValueError("Wrong type of VSF curvatures!")
        
        self.__k = value

class Robot(Frame):
    def __init__(self, id, pose, head, tail, vsf) -> None:
        self.__id = id # Correspond to the model_id
        super().__init__(pose)
        self.__head = head
        self.__tail = tail
        self.__vsf = vsf

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
    def allWheels(self) -> list:
        return self.head.wheels + self.tail.wheels
           