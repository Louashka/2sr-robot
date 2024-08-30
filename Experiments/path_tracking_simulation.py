import os
import glob
import json
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    json_dir = os.path.join(current_dir, 'Data/')

    json_file_names = glob.glob(os.path.join(json_dir, '*tracking_morph_data_2024*.json'))

    for json_file_name in json_file_names:
        json_file = open(json_file_name)
        data = json.load(json_file)

        target_path = np.array([[p['x'], p['y'], p['yaw']] for p in data['path']])
