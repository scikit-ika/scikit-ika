""" An animated visualisation of the WindSimStream, showing how
Pollution emissions and sources change over time and interact with sensors.
"""
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors
import argparse
import numpy as np
import math
import random

from skika.data.synthetic.wind_sim_generator import WindSimGenerator

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-ns", "--nsensor", type=int,
        help="Number of sensors", default=8)
    ap.add_argument("-st", "--sensortype",
        help="How sensors are arranged", default="grid", choices=["circle", "grid"])
    args = vars(ap.parse_args())

    Writer = animation.writers['pillow']
    writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=900)
    fig = plt.figure()
    ax = plt.gca()
    plt.gray()
    n_concepts = 5
    concepts = []
    for c in range(n_concepts):
        concepts.append(np.random.randint(0, 1000))
    current_concept = 0
    stream = WindSimGenerator(produce_image=True, num_sensors= args['nsensor'], sensor_pattern=args['sensortype'])
    stream.prepare_for_use()
    stream.set_concept(current_concept % n_concepts, 1)
    count = 0
    drift_count = 100
    def animate(i):
        global count
        global current_concept
        X, y = stream.next_sample()
        print(f"X: {X}, y: {y}")
        if stream.produce_image:
            z = stream.get_last_image()
            plt.clf()
            plt.imshow(z, norm = matplotlib.colors.Normalize(0, 255))
        count += 1
        if count >= drift_count:
            current_concept += 1
            stream.set_concept(current_concept % n_concepts, 1)
            count = 0
    ani = animation.FuncAnimation(fig, animate, init_func = lambda: [], repeat=True)
    plt.show()