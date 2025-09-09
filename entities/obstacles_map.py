import matplotlib.pyplot as plt
from shapely.geometry import Polygon

class Environment:
    """
    Represents the 2D environment containing obstacles.
    """
    def __init__(self, obstacles: list[Polygon]):
        """
        Initializes the environment.

        Args:
            obstacles (list[Polygon]): A list of shapely Polygon objects 
                                       representing the obstacles.
        """
        if not all(isinstance(obs, Polygon) for obs in obstacles):
            raise TypeError("All obstacles must be shapely.geometry.Polygon objects.")
        
        self.__obstacles = obstacles

    @property
    def obstacles(self) -> list[Polygon]:
        """Returns the list of obstacle polygons."""
        return self.__obstacles

    def plot(self, ax: plt.Axes):
        """
        Plots all obstacles on a given matplotlib axes.

        Args:
            ax (plt.Axes): The axes on which to draw the obstacles.
        """
        for obs in self.__obstacles:
            x, y = obs.exterior.xy
            ax.fill(x, y, alpha=0.7, fc='gray', ec='black')