import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

# Suppose you have a set of 2D points
x = [6, 5.2, 3.5, 2, 2.5]
y = [2, 3.6, 4, 5, 7]

# Fit a spline curve to the data points
tck, _ = splprep([x, y], s=0)

# Calculate the arc length of the spline curve
s = np.linspace(0, 1, 1000)
spline_x, spline_y = splev(s, tck)
spline_arc_length = 0
for i in range(1, len(s)):
    spline_arc_length += np.sqrt((spline_x[i] - spline_x[i-1])**2 + (spline_y[i] - spline_y[i-1])**2)

# Find the midpoint of the spline curve
spline_arc_length_midpoint = spline_arc_length / 2
current_arc_length = 0
for i in range(1, len(s)):
    current_arc_length += np.sqrt((spline_x[i] - spline_x[i-1])**2 + (spline_y[i] - spline_y[i-1])**2)
    if current_arc_length >= spline_arc_length_midpoint:
        midpoint_s = s[i-1]
        midpoint_x, midpoint_y = splev(midpoint_s, tck)
        break

# Calculate the tangent vector at the midpoint
tangent_x, tangent_y = splev(midpoint_s, tck, der=1)
theta = np.arctan2(tangent_y, tangent_x)

# Plot the original data points, the spline curve, and the tangent line
plt.figure(figsize=(8, 6))
plt.scatter(x, y, label='Data points')
plt.plot(spline_x, spline_y, label='Spline curve')
plt.plot([midpoint_x, midpoint_x + 0.5 * np.cos(theta)], [midpoint_y, midpoint_y + 0.5 * np.sin(theta)], color='red', label='Tangent line')
plt.axis('equal')
plt.legend()
plt.show()