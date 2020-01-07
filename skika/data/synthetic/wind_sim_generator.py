import math

import numpy as np
from skmultiflow.utils import check_random_state


class PollutionSource:
    """ Helper class to simulate a source of pollution.

    Parameters
    ----------
    x: int
        X Position of source

    y: int
        Y Position of source

    strength: int
        Amount of pollution emitted

    expansion: int
        How quickly pollution expands

    random_state: int or randomState
        Seed for randomisation
    """

    def __init__(self, x, y, strength, expansion, random_state):
        self.x = x
        self.y = y
        self.strength = strength
        self.expansion = expansion
        self.random_state = check_random_state(random_state)

        self.initial_radius = expansion * 10
        self.last_strength = strength
        self.period = self.random_state.randint(0, 360)
        self.last_emitted = None

    def inside_last_emission(self):
        x_delta = (self.x - self.last_emitted.x)
        y_delta = (self.y - self.last_emitted.y)
        return (x_delta * x_delta + y_delta * y_delta) < self.last_emitted.r

    def emit(self):
        self.period += self.random_state.randint(1, 36)
        s = self.strength * (math.sin(math.radians(self.period)) * 0.1 + 1)
        e = PollutionEmission(
            self.x, self.y, self.initial_radius, s, self.expansion)

        if self.last_emitted is not None and self.inside_last_emission():
            self.last_emitted.child = e
        else:
            e.first_emitted = True
        self.last_emitted = e
        return e


class PollutionEmission:
    """ Helper class to model emitted pollution.

    Parameters
    ----------
    x: int
        X Position of source

    y: int
        Y Position of source

    r: int
        Initial radius

    initial_strength: int
        Amount of pollution emitted initially

    expansion: int
        How quickly pollution expands
    """

    def __init__(self, x, y, r, initial_strength, expansion):
        self.x = x
        self.y = y
        self.r = r
        self.strength = initial_strength

        self.expansion = expansion
        self.diminish = 0.05

        self.alive = True

        self.child = None
        self.first_emitted = False
        self.last_time_step = -1

    def propagate(self, wind_vec, ts):
        if ts == self.last_time_step:
            return
        if self.alive:
            self.x += wind_vec[0]
            self.y += wind_vec[1]

            self.r += self.expansion
            # self.strength -= self.diminish
            self.strength *= (1 - self.diminish)

        if self.strength <= 10:
            self.alive = False
            if self.child is not None:
                self.child.first_emitted = True
        self.last_time_step = ts


class WindSimGenerator:
    """ Generator simulating wood smoke pollution sensors.

    This generator is modeled after experiments placing
    sensors over a town to capture wood pollution readings.

    Pollution sources are distributed and generate an emission
    every few timesteps. Emission movement depends on wind
    speed and direction, as well as internal dissipation.

    Feature sensors pick up readings to generate features,
    while a central target sensor records a target reading.
    This is transformed into a classification target.

    Parameters
    ----------
    concept: int (Default: 0)
        The initial concept

    produce_image: bool (Default: False)
        Whether or not to produce an image for visualisation

    num_sensors: int (Default: 8)
        The number of sensors to place

    sensor_pattern: 'circle', 'grid' (Default: circle)
        The pattern to place sensors

    sample_random_state: int, randomState
        randomState to generate simulation

    x_trail: int (Default: 1)
        The number of feature x lags to output
    """

    def __init__(self, concept=0, produce_image=False, num_sensors=8,
                 sensor_pattern='circle', sample_random_state=None, x_trail=1):
        self.anchor = [0, 0]

        # Treat as meters, i.e 1000 = a 1000x1000 m window
        self.window_width = 200

        # How many grid squares.
        # window_width / window_divisions = size of grid square in meters.
        self.window_divisions = 200
        self.grid_square_width = self.window_width / self.window_divisions

        self.wind_direction = None
        self.concept = concept
        self.set_wind(concept, strength=2.2)

        # Scale of noise, bigger = bigger noise features.
        self.noise_scale = 5000

        center_sensor_loc = (int(self.window_width / 2),
                             int(self.window_width / 2))
        self.optimal_sensor_locs = [center_sensor_loc]
        if sensor_pattern == 'circle':
            radius = self.window_width / 4
            angle = 0
            while angle < 360:
                px = center_sensor_loc[0] + \
                    math.cos(math.radians(angle)) * radius
                py = center_sensor_loc[1] + \
                    math.sin(math.radians(angle)) * radius
                self.optimal_sensor_locs.append((px, py))
                angle += 360 / num_sensors
        else:
            num_sensors_x = math.ceil(math.sqrt(num_sensors))
            num_sensors_y = math.ceil(math.sqrt(num_sensors))
            sensor_x_gap = self.window_width / (num_sensors_x + 1)
            sensor_y_gap = self.window_width / (num_sensors_y + 1)

            for c in range(num_sensors_x):
                for r in range(num_sensors_y):
                    px = (c+1) * sensor_x_gap
                    py = (r+1) * sensor_y_gap
                    self.optimal_sensor_locs.append((px, py))

        self.optimal_sensor_square_locs = []
        for sx, sy in self.optimal_sensor_locs:
            self.optimal_sensor_square_locs.append(
                (int(sx / self.grid_square_width),
                 int(sy / self.grid_square_width)))

        # Timestep in seconds
        self.timestep = 60 * 10

        self.produce_image = produce_image
        self.last_update_image = None

        self.emitted_values = []

        # The number of timesteps a prediction is ahead of X.
        # I.E the y value received with a given X is the y value
        # y_lag ahead of the reveived X values.
        # For this sim, it should be 10 minutes.
        self.y_lag = 1
        self.x_trail = x_trail

        self.prepared = False

        self.world = np.zeros(
            shape=(self.window_width, self.window_width), dtype=float)
        self.sources = []
        self.pollution = []
        self.pollution_chain = []

        self.sample_random_state = check_random_state(sample_random_state)

        self.ex = 0

    def set_concept(self, concept_seed, difficulty=3):
        """ Set windspeed, direction and sources.

        Parameters
        ----------
        concept_seed: int
            Seed for the concept

        difficulty: int (Default: 3)
            How difficult a concept is
        """
        concept_generator = np.random.RandomState(concept_seed)

        self.wind_direction = concept_generator.randint(0, 360)
        self.wind_strength = (
            (concept_generator.rand() * 60) + 60) / (self.window_width / 5)
        wind_direction_corrected = (self.wind_direction - 90) % 360
        self.wind_direction_radians = math.radians(wind_direction_corrected)

        self.wind_strength_x = math.cos(
            self.wind_direction_radians) * self.wind_strength
        self.wind_strength_y = math.sin(
            self.wind_direction_radians) * self.wind_strength

        self.sources = []

        num_sources = concept_generator.randint(difficulty + 2,
                                                difficulty + 10)
        for s in range(num_sources):
            x = concept_generator.randint(self.window_width / 4,
                                          self.window_width / 4 * 2)
            y = concept_generator.randint(self.window_width / 4,
                                          self.window_width / 4 * 2)
            x -= math.cos(self.wind_direction_radians * -1) \
                * (self.window_width / 2)
            y += math.sin(self.wind_direction_radians * -1) \
                * (self.window_width / 2)
            strength = concept_generator.randint(10, 255)
            strength = 170
            size = concept_generator.randint(1, 4)
            self.sources.append(PollutionSource(
                x, y, strength, (self.window_width / 750) * size,
                self.sample_random_state))

    def get_direction_from_concept(self, concept):
        return 45 * concept

    def set_wind(self, concept=0, direc=None, strength=None):
        """ Set wind parameters

        """
        self.concept = concept
        wind_direction = self.get_direction_from_concept(concept)
        if direc is not None:
            wind_direction = direc
        if wind_direction == self.wind_direction:
            return

        # In knots: 1 knot = 0.514 m/s
        # Data average is around 2.2
        self.wind_strength = strength if strength is not None \
            else self.wind_strength
        self.wind_direction = wind_direction % 360

        # Wind direction is a bearing, want a unit circle degree.
        wind_direction_corrected = (self.wind_direction - 90) % 360
        self.wind_direction_radians = math.radians(wind_direction_corrected)

        self.wind_strength_x = math.cos(
            self.wind_direction_radians) * self.wind_strength
        self.wind_strength_y = math.sin(
            self.wind_direction_radians) * self.wind_strength

    def update(self):
        """ Update simulation

        To save time when checking if sensor is within an
        emission, we exclude check for emissions wholely contained
        in anouther emission.
        """
        self.ex += 1
        self.anchor[0] += self.wind_strength_x * 0.514 * self.timestep
        self.anchor[1] += self.wind_strength_y * 0.514 * self.timestep

        alive_p_first = []
        for p in self.pollution_chain:
            p.propagate((self.wind_strength_x, self.wind_strength_y), self.ex)
            if p.alive:
                if p.first_emitted:
                    alive_p_first.append(p)
            while p.child is not None:
                p = p.child
                p.propagate(
                    (self.wind_strength_x, self.wind_strength_y), self.ex)
                if p.alive:

                    if p.first_emitted:
                        alive_p_first.append(p)
        self.pollution_chain = alive_p_first

        if self.ex % 10 == 0:
            for s in self.sources:
                emission = s.emit()
                if emission.first_emitted:
                    self.pollution_chain.append(emission)

        if self.produce_image:
            z = self.get_full_image()
        else:
            z = None

        sensor_windows = []
        for x, y in self.optimal_sensor_locs:
            sensor_collection = []
            sensor_sum = 0
            look_at_sensors = []
            for p in self.pollution_chain:
                node = p
                if ((x - p.x) ** 2) + ((y - p.y) ** 2) > (p.r**2 + 4):
                    continue
                else:
                    keep_looking = True
                    look_at_sensors.append(p)
                    while node.child is not None and keep_looking:
                        node = node.child
                        dist = ((x - node.x) ** 2) + \
                            ((y - node.y) ** 2) > (node.r**2 + 4)
                        if dist:
                            keep_looking = False
                            break
                        look_at_sensors.append(node)

            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    noise_x_pos = int((x + dx))
                    noise_y_pos = int((y + dy))
                    if self.produce_image and False:
                        sensor_collection.append(z[noise_y_pos, noise_x_pos])
                        sensor_sum += z[noise_y_pos, noise_x_pos]
                    else:
                        value = 0
                        for p in look_at_sensors:
                            if ((noise_x_pos - p.x) ** 2) +\
                                    ((noise_y_pos - p.y) ** 2) > p.r**2:
                                continue
                            else:
                                value += p.strength
                        sensor_sum += value
            sensor_windows.append(sensor_sum)

        if self.produce_image:
            for sx, sy in self.optimal_sensor_square_locs:
                sensor_collection = []
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        px = sx + dx
                        py = sy + dy
                        if 0 > px > self.window_width:
                            continue
                        if 0 > py > self.window_width:
                            continue
                        z[sy + dy, sx + dx] = 255

            self.last_update_image = z

        return (list(map(lambda x: x, sensor_windows[1:])), sensor_windows[0])

    def get_last_image(self):
        return self.last_update_image

    def get_full_image(self):
        x_rolls = round(self.wind_strength_x)
        y_rolls = round(self.wind_strength_y)

        self.world = np.roll(self.world, x_rolls, axis=1)
        self.world = np.roll(self.world, y_rolls, axis=0)

        z = np.copy(self.world)
        for p in self.pollution_chain:
            while p is not None:
                for x in range(round(p.x - p.r), round(p.x + p.r)):
                    for y in range(round(p.y - p.r), round(p.y + p.r)):
                        if x < 0 or x >= self.window_width:
                            continue
                        if y < 0 or y >= self.window_width:
                            continue
                        if ((x - p.x) ** 2) + ((y - p.y) ** 2) > p.r**2:
                            continue
                        z[x, y] += p.strength
                p = p.child

        return z

    def add_emissions(self):
        X, y = self.update()
        for index, emit in enumerate([y] + X):
            if index >= len(self.emitted_values):
                self.emitted_values.append([])
            self.emitted_values[index].append(emit)

    def prepare_for_use(self):

        # Need to set up y values for X values y_lag behind.
        print("prepared")
        if self.wind_direction is None:
            self.set_concept(self.concept)
        self.add_emissions()
        for i in range(1 + self.x_trail + self.y_lag + 500):
            self.add_emissions()
        self.X_index = 1 + self.x_trail + 500
        self.y_index = self.X_index + self.y_lag
        self.prepared = True

    def next_sample(self, batch_size=1):
        x_vals = []
        y_vals = []
        for b in range(batch_size):
            self.add_emissions()
            X = []
            for i, x_emissions in enumerate(self.emitted_values[1:]):
                for x_i in range(self.X_index - self.x_trail,
                                 self.X_index + 1):
                    X.append(x_emissions[x_i])
            current_y = self.emitted_values[0][self.y_index]

            last_y = self.emitted_values[0][self.y_index - 1]
            y = 1 if current_y > last_y else 0
            self.X_index += 1
            self.y_index += 1
            x_vals.append(X)
            y_vals.append(y)
        return (np.array(x_vals), np.array(y_vals))

    def get_info(self, concept=None, strength=None):
        c = concept if concept is not None else self.concept
        s = strength if strength is not None else self.wind_strength
        return f"Direction: {self.get_direction_from_concept(c)}, Speed: {s}"

    def n_remaining_samples(self):
        return -1

    def has_more_samples(self):
        return True
