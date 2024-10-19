from gekko import GEKKO
import numpy as np
from Model import global_var as gv

class Transporter:
    def __init__(self) -> None:
        self.T = 11
        
        self.m = GEKKO(remote=False)
        self.m.time = np.linspace(0, gv.DT * (self.T-1), self.T)