import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk

root = tk.Tk()
frame = tk.Frame(root)
fig, ax = plt.subplots()
canvas = FigureCanvasTkAgg(fig, master = frame)