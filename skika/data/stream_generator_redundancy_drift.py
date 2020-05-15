from skmultiflow.data.base_stream import Stream
#from skmultiflow.data import pseudo_random_processes as prp
from skmultiflow.utils import check_random_state

import copy
#import numpy as np
# import random
#import math

from skika.data.random_rbf_generator_redund import RandomRBFGeneratorRedund


class StreamGeneratorRedund(Stream):
    """ Stream generator with change in number of redundant features
        
        Create a stream from RandomRBFRedun or HyperPlanRedun to generate a given 
        number of drifts with a given number of instances. Each concept contains a 
        different number of redundant features (0, 10, 20, 30, 40, 50, 60, 70, 80, 90 or 100% 
        of the total number of features).
        
        Drifts are regularly placed every n_instances/n_drifts instances. 
        
        Parameters
        ----------
        base_stream: Stream (Default: RandomRBFRedun)
            The base stream to use.
        
        random_state: int, RandomState instance or None, optional (default=None)
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used
            by `np.random`.
        
        n_drifts: int (Default: 10)
            Number of drifts to be generated. 
        
        n_instances: int (Default: 10000)
            Number of instances to be generated. 
        
        Example
        --------
        >>> # Imports
        >>> from skika.data.stream_generator_redundancy_drift import StreamGeneratorRedund
        >>> from skika.data.random_rbf_generator_redund import RandomRBFGeneratorRedund
        >>> # Set the stream
        >>> stream = StreamGeneratorRedund(base_stream = RandomRBFGeneratorRedund(n_classes=2, n_features=30, n_centroids=50, noise_percentage = 0.0), random_state=None, n_drifts = 10, n_instances = 10000)
        >>> stream.prepare_for_use()
        >>> # Retrieve next sample
        >>> stream.next_sample()
        (array([[0.21780997, 0.37810599, 0.24129934, 0.78979064, 0.83463727,
                     0.90272964, 0.5611584 , 0.58977699, 0.78035701, 0.89178544,
                     0.55418949, 0.30293076, 0.09691338, 0.75894948, 0.03441104,
                     0.58977699, 0.75894948, 0.24129934, 0.78979064, 0.83463727,
                     0.37810599, 0.55418949, 0.75894948, 0.24129934, 0.55418949,
                     0.78035701, 0.09691338, 0.90272964, 0.83463727, 0.24129934]]),
        array([1]))
        

    """

    def __init__(self, base_stream = RandomRBFGeneratorRedund(n_classes=2, n_features=30, n_centroids=50, noise_percentage = 0.0), random_state=None, n_drifts = 10, n_instances = 10000):
        super().__init__()
        self.base_stream = base_stream
        self.random_state = random_state
        self._random_state = None # This is the actual random_state object used internally
        self.n_drifts = n_drifts
        self.n_instances = n_instances
        self.name = "Stream Generator Redundancy"

        self.perc_redund_features = self.base_stream.perc_redund_features
        self.n_targets = self.base_stream.n_targets
        self.n_features = self.base_stream.n_features
        self.n_num_features = self.base_stream.n_num_features
        self.n_classes = self.base_stream.n_classes
        self.feature_names = self.base_stream.feature_names
        self.target_names = self.base_stream.target_names
        self.target_values = self.base_stream.target_values

        self.__configure()

    @property
    def perc_redund_features(self):
        """ Retrieve the number of redundant features.
        Returns
        -------
        int
            The total number of redundant features.
        """
        return self._perc_redund_features

    @perc_redund_features.setter
    def perc_redund_features(self, perc_redund_features):
        """ Set the number of redundant features

        """
        self._perc_redund_features = perc_redund_features

    def __configure(self):

        self.list_perc_redund = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

    def prepare_for_use(self):
        """ Prepares the stream for use.
        Randomly create the list of redundant numbers of features used for each concept in the stream.
        
        Notes
        -----
        This functions should always be called after the stream initialization.
        """
        self._random_state = check_random_state(self.random_state)

        # Generate a random list of n_drifts+1 percentages of redundancy to create n_drifts+1 RBF streams
        # Create n_drifts+1 streams to cope with n_drfits
        self.list_perc_redund_drifts = []
        for i in range(self.n_drifts+1):
            if i == 0 :
                perc = self._random_state.choice(self.list_perc_redund)
                self.list_perc_redund_drifts.append(perc)
            else :
                perc = self._random_state.choice(self.list_perc_redund)

                while perc == self.list_perc_redund_drifts[i-1]: # To avoid having two times in a row the same percentage
                    perc = self._random_state.choice(self.list_perc_redund)

                self.list_perc_redund_drifts.append(perc)

        self.list_Streams = []
        for i in range(self.n_drifts+1):
            # # TODO: use self.base_stream to be able to change the type of stream in the future
            self.base_stream.perc_redund_features = self.list_perc_redund_drifts[i]
            self.list_Streams.append(copy.copy(self.base_stream))

        # Generate a random list of n_drifts streams points where drift happens
        self.list_drift_points = []
        for i in range(self.n_drifts):
            self.list_drift_points.append(int(self._random_state.rand()*self.n_instances))
        self.list_drift_points.sort()

        self.current_count = 0
        self.current_concept = 0
        self.current_stream = self.list_Streams[self.current_concept]
        self.current_stream.prepare_for_use()
        self.perc_redund_features = self.list_Streams[self.current_concept].perc_redund_features


    def next_sample(self, batch_size=1):

        """ Return batch_size samples generated by choosing a centroid at
        random and randomly offsetting its attributes so that it is
        placed inside the hypersphere of that centroid.

        Parameters
        ----------
        batch_size: int
            The number of samples to return.

        Returns
        -------
        tuple or tuple list
            Return a tuple with the features matrix and the labels matrix for
            the batch_size samples that were requested.

        """

        current_sample_x, current_sample_y = self.current_stream.next_sample(batch_size)

        if self.current_count in self.list_drift_points :
            self.current_concept = self.list_drift_points.index(self.current_count)+1
            self.current_stream = self.list_Streams[self.current_concept]
            self.current_stream.prepare_for_use()
            self.perc_redund_features = self.list_Streams[self.current_concept].perc_redund_features

        self.current_count = self.current_count + batch_size

        return current_sample_x, current_sample_y

    def has_more_samples(self):
            """
            Checks if stream has more samples.
            Returns
            -------
            Boolean
                True if stream has more samples.
            """
            if self.n_instances > self.current_count :
                return True
            else :
                return False
