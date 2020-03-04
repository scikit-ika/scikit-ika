import math
import numpy as np
import logging

from skmultiflow.utils import check_random_state
from skmultiflow.data import AGRAWALGenerator
from skmultiflow.data import SEAGenerator
from skmultiflow.data import SineGenerator
from skmultiflow.data import STAGGERGenerator
from skmultiflow.data import LEDGeneratorDrift
from skmultiflow.data import MIXEDGenerator
from skmultiflow.data import HyperplaneGenerator
from skmultiflow.data import ConceptDriftStream

class RecurrentDriftStream(ConceptDriftStream):

    """ Generates a stream with recurrent concept drifts.

    Parameters
    __________
    generator:
        a stream generator

    """

    def __init__(self,
                 generator='agrawal',
                 stable_period=3000,
                 position=3000,
                 concepts=[4, 0, 8],
                 width=1,
                 lam=1.0,
                 has_noise=False,
                 all_concepts=[4, 0, 8, 6, 2, 1, 3, 5, 7, 9],
                 concept_shift_step=-1,
                 concept_shift_sample_intervals=[200000, 250000, 300000],
                 stable_period_lam=-1,
                 stable_period_start=1000,
                 stable_period_base=200,
                 stable_period_logger=None,
                 random_state=0):

        super().__init__()

        self.streams = []
        self.cur_stream = None
        self.stream_idx = 0
        self.drift_stream_idx = 0
        self.sample_idx = 0

        self.generator = generator
        self.stable_period = stable_period
        self.position = position
        self.concepts = concepts
        self.random_state = random_state
        self._random_state = check_random_state(self.random_state)
        self.width = width

        self.lam = lam
        self.concepts_probs = []

        self.has_noise = has_noise
        self.noises = [0.1, 0.2, 0.3, 0.4]
        self.noise_probs = self.__get_poisson_probs(4, self.lam)

        self.stable_period_lam = stable_period_lam
        self.stable_period_start = stable_period_start
        self.stable_period_base  = stable_period_base
        self.stable_period_probs = \
                self.__get_poisson_probs(20, self.stable_period_lam)
        self.stable_period_logger = stable_period_logger
        print(f"stable_period_probs: {self.stable_period_probs}")

        self.concept_shift_step = concept_shift_step
        self.concept_shift_sample_intervals = concept_shift_sample_intervals
        self.all_concepts = all_concepts
        self.total_sample_idx = 0

    def next_sample(self, batch_size=1):

        """ Returns the next `batch_size` samples.

        Parameters
        ----------
        batch_size: int
            The number of samples to return.

        Returns
        -------
        tuple or tuple list
            Return a tuple with the features matrix
            for the batch_size samples that were requested.

        """
        self.current_sample_x = np.zeros((batch_size, self.n_features))
        self.current_sample_y = np.zeros((batch_size, self.n_targets))

        for j in range(batch_size):
            self.sample_idx += 1
            x = -4.0 * float(self.sample_idx - self.position) / float(self.width)
            probability_drift = 1.0 / (1.0 + np.exp(x))

            if self._random_state.rand() > probability_drift:
                X, y = self.cur_stream.next_sample()
            else:
                X, y = self.drift_stream.next_sample()
            self.current_sample_x[j, :] = X
            self.current_sample_y[j, :] = y

        if self.sample_idx >= self.stable_period + self.width:
            self.sample_idx = 0

            # strict cyclic
            # self.stream_idx = (self.stream_idx + 1) % len(self.streams)
            # self.drift_stream_idx = (self.stream_idx + 1) % len(self.streams)

            # finite poisson
            self.stream_idx = self.drift_stream_idx
            self.drift_stream_idx = self.__get_next_random_idx(self.concept_probs)

            self.cur_stream = self.streams[self.stream_idx]
            self.drift_stream = self.streams[self.drift_stream_idx]

            if self.stable_period_lam > 0:
                self.stable_period = self.stable_period_start \
                        + self.stable_period_base \
                        * self.__get_next_random_idx(self.stable_period_probs)
                self.position = self.stable_period
                print(self.position)

                if self.stable_period_logger is not None:
                    self.stable_period_logger.info(str(self.stable_period))

            # generate random noise
            if self.has_noise and self.generator == 'agrawal':
                self.noise_idx = self.__get_next_random_idx(self.noise_probs)
                self.cur_stream.perturbation = self.noises[self.noise_idx]

        self.total_sample_idx += batch_size
        self.__concept_shift()

        return self.current_sample_x, self.current_sample_y.flatten()

    def get_data_info(self):
        return self.cur_stream.get_data_info()

    def prepare_for_use(self):
        if self.generator in ['sea', 'sine']:
            self.concepts = [v for v in range(0, 4)]
        elif self.generator in ['stagger']:
            self.concepts = [v for v in range(0, 3)]
        elif self.generator in ['mixed']:
            self.concepts = [v for v in range(0, 2)]
        elif self.generator in ['led']:
            self.concepts = [v for v in range(0, 7)]

        if self.concept_shift_step > 0:
            for concept in self.all_concepts:
                stream = AGRAWALGenerator(classification_function=concept,
                                          random_state=self.random_state,
                                          balance_classes=False,
                                          perturbation=0.05)
                stream.prepare_for_use()
                self.streams.append(stream)
        else:

            for concept in self.concepts:
                if self.generator == 'agrawal':
                    stream = AGRAWALGenerator(classification_function=concept,
                                              random_state=self.random_state,
                                              balance_classes=False,
                                              perturbation=0.05)
                elif self.generator == 'sea':
                    stream = SEAGenerator(classification_function=concept,
                                          random_state=self.random_state,
                                          balance_classes=False,
                                          noise_percentage=0.05)
                elif self.generator == 'sine':
                    stream = SineGenerator(classification_function=concept,
                                           random_state=self.random_state,
                                           balance_classes=False,
                                           has_noise=False)
                elif self.generator == 'stagger':
                    stream = STAGGERGenerator(classification_function=concept,
                                              random_state=self.random_state,
                                              balance_classes=False)
                elif self.generator == 'mixed':
                    stream = MIXEDGenerator(classification_function=concept,
                                            random_state=self.random_state,
                                            balance_classes=False)
                elif self.generator == 'led':
                    stream = LEDGeneratorDrift(random_state=self.random_state,
                                               has_noise=True,
                                               n_drift_features=concept)
                else:
                    print(f"unknown stream generator {self.generator}")
                    exit()

                stream.prepare_for_use()
                self.streams.append(stream)

        self.cur_stream = self.streams[0]
        self.drift_stream = self.streams[1]

        stream = self.cur_stream
        self.n_samples = stream.n_samples
        self.n_targets = stream.n_targets
        self.n_features = stream.n_features
        self.n_num_features = stream.n_num_features
        self.n_cat_features = stream.n_cat_features
        self.n_classes = stream.n_classes
        self.cat_features_idx = stream.cat_features_idx
        self.feature_names = stream.feature_names
        self.target_names = stream.target_names
        self.target_values = stream.target_values
        self.n_targets = stream.n_targets
        self.name = 'drifting' + stream.name

        print(f"len: {len(self.concepts)}")
        self.concept_probs = \
                self.__get_poisson_probs(len(self.concepts), self.lam)

    def __get_next_random_idx(self, probs):
        # r = random.uniform(0, 1)
        r = self._random_state.uniform(0, 1)
        # print(f"next_random_val={r}")
        cur_sum = 0

        for idx, val in enumerate(probs):
            cur_sum += val
            if r < cur_sum:
                return idx

    def __get_poisson_probs(self, num_events, lam):
        probs = [0] * num_events
        for e in range(0, num_events):
            probs[e] = self.__calc_poisson_prob(e, lam)

        norm_probs = [float(i)/sum(probs) for i in probs]
        print(f"norm_probs: {norm_probs}")
        return norm_probs

    def __calc_poisson_prob(self, k, lam):
        return math.pow(lam, k) * math.exp(-lam) / math.factorial(k)

    def __concept_shift(self):
        if self.concept_shift_step <= 0:
            return

        if not self.total_sample_idx == 0 \
                and self.total_sample_idx % self.concept_shift_sample_intervals[0] == 0:
            interval = self.concept_shift_sample_intervals.pop(0)
            self.concept_shift_sample_intervals.append(interval)

        else:
            return

        # concept shift
        for i in range(self.concept_shift_step):
            stream = self.streams.pop(0)
            self.streams.append(stream)
