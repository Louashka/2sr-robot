import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as PlotPolygon

edge_color = '#acb1c4'
passage_color = '#79addc'

with open('experiments/figures/voronoi_plot_0.pkl', 'rb') as f:
    voronoi_data = pickle.load(f)

    fig = voronoi_data['figure']
    ax = voronoi_data['axes']

    print("Extracting data from ax.lines:")
    extracted_lines = {}
    for i, line in enumerate(ax.lines):
        x_data = line.get_xdata()
        y_data = line.get_ydata()
        label = line.get_label()
        print(f"  - Line {i} (Label: '{label}') has {len(x_data)} points.")
        extracted_lines[label] = {'x': x_data, 'y': y_data}

    print("\nExtracting data from ax.collections:")
    extracted_polygons_vertices = []

    # 1. Iterate through the collections on the axes
    #    (There might be others from scatter plots, etc.)
    for collection in ax.collections:
        
        # Optional: Check if it's the right type, though often you'll only have one.
        if isinstance(collection, PatchCollection):
            
            # 2. Use get_paths() to get a list of Path objects
            paths = collection.get_paths()
            
            print(f"Found a PatchCollection with {len(paths)} polygons.")
            
            # 3. Iterate through the Path objects and get the vertices
            for path in paths:
                # The .vertices attribute gives us the (N, 2) NumPy array of coordinates
                vertices = path.vertices
                extracted_polygons_vertices.append(vertices)
                
                print(f"  - Extracted a polygon with {len(vertices)} vertices.")


    plt.show()


