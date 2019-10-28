"""
Show an animated visual of windSimStream, showing which sensor is most predictive.
"""
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors
import argparse
import numpy as np
import math
import random

from skika.data.synthetic.wind_sim_generator import WindSimGenerator

datastream = WindSimGenerator(produce_image=True)
datastream.prepare_for_use()
datastream.set_concept(0, 1)
Writer = animation.writers['pillow']
writer = Writer(fps=60, metadata=dict(artist='Me'), bitrate=1800)
fig = plt.figure()
ax = plt.gca()
plt.gray()

last_sensor_windows = None
count = 0

def animate(i):
    global count
    global datastream
    count += 1
    print(count)
    if count > 100:
        count = 0
        concept = random.randint(0, 8)
        datastream.set_concept(concept, 1)

    closest_match = None
    closest_distance = None
    X, y = datastream.next_sample()
    X = X[0]
    y = y[0]
    z = datastream.get_last_image()
    for i,sv in enumerate([x for i,x in enumerate(X) if (i%2 - 1) != 0]):
        distance = sv - y
        print(f"{i}: {distance}")
        if closest_distance == None or distance < closest_distance:
            closest_distance = distance
            closest_match = i + 1
    print(f"Closest match to center was {closest_match}")
    if not z is None:
        for si, l in enumerate(datastream.optimal_sensor_square_locs):
            sx, sy = l
            if si in [closest_match, 0]:
                for dx in range(-10, 11):
                    for dy in range(-10, 11):
                        z[sy + dy, sx + dx] = 255
        # Plot the grid
        plt.clf()
        plt.imshow(z)
ani = animation.FuncAnimation(fig, animate, init_func = lambda: [], repeat=True)
plt.show()