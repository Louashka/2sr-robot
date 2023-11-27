import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk

def createWindow():
    window = tk.Tk()
    frame = tk.Frame(window)
    fig, ax = plt.subplots()
    canvas = FigureCanvasTkAgg(fig, master = frame)

    window.title('Robot motion')
    window.config(background='#fafafa')

    canvas.get_tk_widget().pack()
    frame.pack()

    return window

def plotMotion(mas, manipulandums):
    pass