import os
import pandas as pd
import numpy as np

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'Data/vss_log2.csv')

print(dirname)

df = pd.read_csv(filename, names=["stiffness", "T", "mode", "time"])

print(df)