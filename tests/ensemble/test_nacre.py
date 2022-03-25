import copy
from collections import deque
import math
from random import randrange
import time

import numpy as np
from sklearn.metrics import cohen_kappa_score
from skika.nacre import nacre_wrapper

expected_accuracies = \
    [0.56, 0.492, 0.45, 0.71, 0.615, 0.436, 0.39, 0.481, 0.488, 0.754]

def test_nacre():

    # classification settings
    np.random.seed(0)
    stream_file_path = "./recurrent-data/real-world/covtype.arff"
    max_samples = 10000
    sample_freq = 1000

    # pearl specific params
    num_trees = 60
    max_num_candidate_trees = 120
    repo_size = 9000
    edit_distance_threshold = 90
    kappa_window = 50
    lossy_window_size = 100000000
    reuse_window_size = 0
    max_features = -1
    bg_kappa_threshold = 0
    cd_kappa_threshold = 0.4
    reuse_rate_upper_bound = 0.18
    warning_delta = 0.0001
    drift_delta = 0.00001

    poisson_lambda = 6
    random_state = 0

    # nacre specific params
    grpc_port = 50051 # port number for the sequence prediction grpc service
    pro_drift_window = 100 # number of instances must be seen for proactive drift adaptation
    seq_len = 8 # sequence length for sequence predictor
    backtrack_window = 25 # number of instances per eval when backtracking
    stability_delta = 0.001
    hybrid_delta = 0.001

    classifier = nacre_wrapper.nacre_wrapper(
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
                     seq_len=8)


    correct = 0
    window_actual_labels = []
    window_predicted_labels = []

    start_time = time.process_time()

    for count in range(0, max_samples):
        if not classifier.get_next_instance():
            break

        # test
        prediction = classifier.predict()

        actual_label = classifier.get_cur_instance_label()
        if prediction == actual_label:
            correct += 1

        window_actual_labels.append(actual_label)
        window_predicted_labels.append(prediction)

        classifier.train()

        # log performance
        if count % sample_freq == 0 and count != 0:
            accuracy = correct / sample_freq
            assert accuracy == expected_accuracies[int(count/sample_freq) - 1]
            # elapsed_time = time.process_time() - start_time + classifier.cpt_runtime
            # kappa = cohen_kappa_score(window_actual_labels, window_predicted_labels)
            # candidate_tree_size = classifier.get_candidate_tree_group_size()
            # tree_pool_size = classifier.get_tree_pool_size()

            # print(f"{count},{accuracy},{kappa},{candidate_tree_size},{tree_pool_size},{str(elapsed_time)}")

            correct = 0
            window_actual_labels = []
            window_predicted_labels = []
