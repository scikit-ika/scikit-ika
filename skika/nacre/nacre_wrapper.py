import copy
from collections import deque
import math
from random import randrange
import time

import numpy as np
import grpc
from . import seqprediction_pb2
from . import seqprediction_pb2_grpc

from skika.ensemble import nacre
from skika.third_party.denstream.DenStream import DenStream



class nacre_wrapper:
    """
    Description :
      Proactive Recurrent Drift Classifier

    """

    def __init__(self,
                 grpc_port=50051,
                 seed=0,
                 stream_file_path="./recurrent-data/real-world/covtype.arff",
                 num_trees=60,
                 max_num_candidate_trees=120,
                 repo_size=9000,
                 edit_distance_threshold=90,
                 kappa_window=50,
                 lossy_window_size=100000000,
                 reuse_window_size=0,
                 max_features=-1,
                 bg_kappa_threshold=0,
                 cd_kappa_threshold=0.4,
                 reuse_rate_upper_bound=0.18,
                 warning_delta=0.0001,
                 drift_delta=0.00001,
                 poisson_lambda=6,
                 random_state=0,
                 pro_drift_window=100,
                 backtrack_window=25,
                 stability_delta=0.001,
                 hybrid_delta=0.001,
                 seq_len=8):


      np.random.seed(seed)

      self.num_trees = num_trees

      self.pro_drift_window = pro_drift_window
      self.seq_len = seq_len
      self.hybrid_delta = hybrid_delta

      self.num_request = 0
      self.cpt_runtime = 0
      self.num_instances_seen = 0

      self.next_adapt_state_locs = [-1 for v in range(num_trees)]
      self.predicted_drift_locs = [-1 for v in range(num_trees)]
      self.drift_interval_sequences = [deque(maxlen=seq_len) for v in range(num_trees)]
      self.last_actual_drift_points = [0 for v in range(num_trees)]


      self.clusterer = DenStream(lambd=0.1, eps=10, beta=0.5, mu=3)
      self.classifier = nacre(
                         num_trees,
                         max_num_candidate_trees,
                         repo_size,
                         edit_distance_threshold,
                         kappa_window,
                         lossy_window_size,
                         reuse_window_size,
                         max_features,
                         poisson_lambda,
                         random_state,
                         bg_kappa_threshold,
                         cd_kappa_threshold,
                         reuse_rate_upper_bound,
                         warning_delta,
                         drift_delta,
                         pro_drift_window,
                         hybrid_delta,
                         backtrack_window,
                         stability_delta)
      self.classifier.init_data_source(stream_file_path);

      self.stub = None
      self._init_channel(grpc_port)

    def _init_channel(self, grpc_port):
        channel = grpc.insecure_channel(f'localhost:{grpc_port}')
        print(f'Sequence prediction server is listening at {grpc_port}...')
        self.stub = seqprediction_pb2_grpc.PredictorStub(channel)
        self.stub.setNumTrees(seqprediction_pb2.SetNumTreesMessage(numTrees=self.num_trees))

    def _fit_predict(self, clusterer, interval):
        x = [np.array([interval])]
        label = clusterer.fit_predict(x)[0]

        if label == -1:
            return interval

        return int(round(clusterer.p_micro_clusters[label].center()[0]))

    def get_tree_pool_size(self):
        return self.classifier.get_tree_pool_size()

    def get_candidate_tree_group_size(self):
        return self.classifier.get_candidate_tree_group_size()

    def get_next_instance(self):
        return self.classifier.get_next_instance()

    def get_cur_instance_label(self):
        return self.classifier.get_cur_instance_label()

    def predict(self):
        return self.classifier.predict()

    def train(self):
        self.classifier.train()
        # classifier.delete_cur_instance()

        # Generate new sequences for the actual drifted trees
        for idx in self.classifier.get_stable_tree_indices():

            # find actual drift point at num_instances_before
            num_instances_before = self.classifier.find_last_actual_drift_point(idx)

            if num_instances_before > -1:
                interval = self.num_instances_seen - num_instances_before - self.last_actual_drift_points[idx]
                if interval < 0:
                    pass
                    # print("Failed to find the actual drift point")
                else:
                    interval = self._fit_predict(self.clusterer, interval)
                    self.drift_interval_sequences[idx].append(interval)
                    self.last_actual_drift_points[idx] = self.num_instances_seen - num_instances_before
            else:
                continue

            # train CPT with the new sequence
            if len(self.drift_interval_sequences[idx]) >= self.seq_len:
                self.num_request += 1
                cpt_response = self.stub.train(seqprediction_pb2
                                  .SequenceMessage(seqId=self.num_request,
                                                   treeId=idx,
                                                   seq=self.drift_interval_sequences[idx]))
                if cpt_response.result:
                    self.cpt_runtime += cpt_response.runtimeInSeconds
                else:
                    print("CPT training failed")
                    exit()

            # predict the next drift points
            if len(self.drift_interval_sequences[idx]) >= self.seq_len:
                self.drift_interval_sequences[idx].popleft()

                response = self.stub.predict(seqprediction_pb2
                                            .SequenceMessage(seqId=self.num_instances_seen,
                                                             treeId=idx,
                                                             seq=self.drift_interval_sequences[idx]))
                self.cpt_runtime += cpt_response.runtimeInSeconds

                if len(response.seq) > 0:
                    interval = response.seq[0]

                    self.predicted_drift_locs[idx] = self.last_actual_drift_points[idx] + interval
                    self.next_adapt_state_locs[idx] = self.last_actual_drift_points[idx] + interval \
                                                 + self.pro_drift_window + 1

                    self.drift_interval_sequences[idx].append(interval)

        # check if hit predicted drift locations
        transition_tree_pos_list = []
        adapt_state_tree_pos_list = []

        for idx in range(self.num_trees):
            # find potential drift trees for candidate selection
            if self.num_instances_seen >= self.predicted_drift_locs[idx] and self.predicted_drift_locs[idx] != -1:
                self.predicted_drift_locs[idx] = -1
                # TODO if not classifier.actual_drifted_trees[idx]:
                transition_tree_pos_list.append(idx)

            # find trees with actual drifts
            if self.num_instances_seen >= self.next_adapt_state_locs[idx] and self.next_adapt_state_locs[idx] != -1:
                self.next_adapt_state_locs[idx] = -1
                if self.classifier.has_actual_drift(idx):
                    adapt_state_tree_pos_list.append(idx)

        if len(transition_tree_pos_list) > 0:
            # select candidate_trees
            self.classifier.select_candidate_trees(transition_tree_pos_list)
        if len(adapt_state_tree_pos_list) > 0:
            # update actual drifted trees

            actual_drifted_tree_indices = \
                self.classifier.adapt_state(adapt_state_tree_pos_list, False)
            print(f"LOG: actual_drifted_tree_indices: {actual_drifted_tree_indices}")

        self.num_instances_seen += 1
