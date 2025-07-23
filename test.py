# import matplotlib.pyplot as plt
# import numpy as np
# import matplotlib.animation as animation

# fig, ax = plt.subplots()
# x = np.arange(0, 2 * np.pi, 0.01)
# line, = ax.plot(x, np.sin(x))

# def animate(i):
#     # Update the data
#     line.set_ydata(np.sin(x + i / 50))

#     # Change the color based on a condition or frame number
#     if i % 10 == 0:  # Change color every 10 frames
#         line.set_color(np.random.rand(3,)) # Random RGB color
#     elif i % 5 == 0:
#         line.set_color('red')
#     else:
#         line.set_color('blue')

#     return line, # Return the modified artist(s)

# ani = animation.FuncAnimation(fig, animate, interval=20, blit=True)
# plt.show()

from collection import motive_client

if __name__ == "__main__":
    mocap = motive_client.MocapReader()
    mocap.startDataListener()

    while True:
        print(mocap.data)
        if mocap.data:
            print(mocap.data.get_as_string())