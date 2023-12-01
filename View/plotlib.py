import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
import tkinter.font as font
import numpy as np


class GUI:
    def __init__(self, task) -> None:
        self.__task = task

        self.__window = None
        self.__canvas = None
        self.__ax = None

        self.__createWindow()
        self.__createSideBar()
        self.__defineTrackingArea()

    @property
    def window(self) -> object:
        return self.__window

    def __createWindow(self) -> None:
        self.__window = tk.Tk()

        self.__window.title('Robot motion')
        self.__window.config(background='#fafafa')
        self.__window.geometry('1200x720')        

    def __createSideBar(self) -> None:
        self.__side_bar_frame = tk.Frame(self.__window)
        self.__side_bar_frame.pack(fill=tk.Y, side=tk.RIGHT)

        self.__buttons_frame = tk.Frame(self.__side_bar_frame)
        self.__buttons_frame.pack(fill=tk.NONE, expand=True)

        self.__buttons = {}
        
        self.__buttons['gen_path'] = tk.Button(self.__buttons_frame, text='Generate path', command=self.__task.generatePath)
        self.__buttons['start'] = tk.Button(self.__buttons_frame, text='Start')
        self.__buttons['stop'] = tk.Button(self.__buttons_frame, text='Stop')
        self.__buttons['exit'] = tk.Button(self.__buttons_frame, text='Exit')
        
        for button in self.__buttons.values():
            button['font'] = font.Font(size=26)
            button.pack(fill=tk.X, padx=20, pady=8)

    def __defineTrackingArea(self) -> None:
        self.__masFrame = tk.Frame(self.__window)
        self.__masFrame.pack(expand=True)
        self.__fig, self.__ax = plt.subplots()
        self.__canvas = FigureCanvasTkAgg(self.__fig, master = self.__masFrame)
        self.__canvas.get_tk_widget().pack()

    def plotPaths(self, paths, x_lim, y_lim) -> None:
        self.__ax.clear()

        for path in paths:
            path = np.array(path)
            path_x = path[:,0]
            path_y = path[:,1]

            self.__ax.plot(path_x, path_y)
              
        # self.__ax.axis('equal')
        self.__ax.set_xlim(x_lim)
        self.__ax.set_ylim(y_lim)

        plt.tight_layout()

        self.__canvas.draw()