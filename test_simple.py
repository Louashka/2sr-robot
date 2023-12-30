import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    p = [[1, 2], [-3, 8], [2, 2], [6, 0]]

    p_array = np.array(p).T - np.array([[1, 2]]).T
    print(p_array.tolist())

