import copy
from collections import deque
import math
from random import randrange
import time

import numpy as np
from sklearn.metrics import cohen_kappa_score
from skika.ensemble import adaptive_random_forest

import sys
paths = [r'..', r'../third_party'] # to locate denstream
for path in paths:
    if path not in sys.path:
        sys.path.append(path)

def test_nacre():

    # classification settings
    np.random.seed(0)
    stream_file_path = "./recurrent-data/real-world/covtype.arff"
    max_samples = 200000
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
    enable_state_adaption = True
    enable_state_graph = True

    poisson_lambda = 6
    random_state = 0

    # nacre specific params
    grpc_port = 50051 # port number for the sequence prediction grpc service
    pro_drift_window = 100 # number of instances must be seen for proactive drift adaptation
    seq_len = 8 # sequence length for sequence predictor
    backtrack_window = 25 # number of instances per eval when backtracking
    stability_delta = 0.001
    hybrid_delta = 0.001

    classifier = nacre(num_trees,
                     max_num_candidate_trees,
                     repo_size,
                     edit_distance_threshold,
                     kappa_window,
                     lossy_window_size,
                     reuse_window_size,
                     arf_max_features,
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

    # TODO move nacre's algorithm to skika/ensemble

    import grpc
    import seqprediction_pb2
    import seqprediction_pb2_grpc

    from denstream.DenStream import DenStream

    clusterer = DenStream(lambd=0.1, eps=10, beta=0.5, mu=3)

    def fit_predict(clusterer, interval):
        x = [np.array([interval])]
        label = clusterer.fit_predict(x)[0]

        if label == -1:
            return interval

        return int(round(clusterer.p_micro_clusters[label].center()[0]))

    correct = 0
    window_actual_labels = []
    window_predicted_labels = []

    num_request = 0
    cpt_runtime = 0

    next_adapt_state_locs = [-1 for v in range(num_trees)]
    predicted_drift_locs = [-1 for v in range(num_trees)]
    drift_interval_sequences = [deque(maxlen=seq_len) for v in range(num_trees)]
    last_actual_drift_points = [0 for v in range(num_trees)]

    start_time = time.process_time()

    classifier.init_data_source(stream_file_path);

    with grpc.insecure_channel(f'localhost:{grpc_port}') as channel:
        print(f'Sequence prediction server is listening at {grpc_port}...')

        stub = seqprediction_pb2_grpc.PredictorStub(channel)

        stub.setNumTrees(seqprediction_pb2.SetNumTreesMessage(numTrees=num_trees))

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

            # train
            classifier.train()
            # TODO
            # classifier.delete_cur_instance()

            # Generate new sequences for the actual drifted trees
            for idx in classifier.get_stable_tree_indices():

                # find actual drift point at num_instances_before
                num_instances_before = classifier.find_last_actual_drift_point(idx)

                if num_instances_before > -1:
                    interval = count - num_instances_before - last_actual_drift_points[idx]
                    if interval < 0:
                        print("Failed to find the actual drift point")
                        # exit()
                    else:
                        interval = fit_predict(clusterer, interval)
                        drift_interval_sequences[idx].append(interval)
                        last_actual_drift_points[idx] = count - num_instances_before
                else:
                    continue

                # train CPT with the new sequence
                if len(drift_interval_sequences[idx]) >= seq_len:
                    num_request += 1
                    cpt_response = stub.train(seqprediction_pb2
                                      .SequenceMessage(seqId=num_request,
                                                       treeId=idx,
                                                       seq=drift_interval_sequences[idx]))
                    if cpt_response.result:
                        cpt_runtime += cpt_response.runtimeInSeconds
                    else:
                        print("CPT training failed")
                        exit()

                # predict the next drift points
                if len(drift_interval_sequences[idx]) >= seq_len:
                    drift_interval_sequences[idx].popleft()

                    response = stub.predict(seqprediction_pb2
                                                .SequenceMessage(seqId=count,
                                                                 treeId=idx,
                                                                 seq=drift_interval_sequences[idx]))
                    cpt_runtime += cpt_response.runtimeInSeconds

                    if len(response.seq) > 0:
                        interval = response.seq[0]

                        predicted_drift_locs[idx] = last_actual_drift_points[idx] + interval
                        next_adapt_state_locs[idx] = last_actual_drift_points[idx] + interval \
                                                     + pro_drift_window + 1

                        drift_interval_sequences[idx].append(interval)

            # check if hit predicted drift locations
            transition_tree_pos_list = []
            adapt_state_tree_pos_list = []

            for idx in range(num_trees):
                # find potential drift trees for candidate selection
                if count >= predicted_drift_locs[idx] and predicted_drift_locs[idx] != -1:
                    predicted_drift_locs[idx] = -1
                    # TODO if not classifier.actual_drifted_trees[idx]:
                    transition_tree_pos_list.append(idx)

                # find trees with actual drifts
                if count >= next_adapt_state_locs[idx] and next_adapt_state_locs[idx] != -1:
                    next_adapt_state_locs[idx] = -1
                    if classifier.has_actual_drift(idx):
                        adapt_state_tree_pos_list.append(idx)

            if len(transition_tree_pos_list) > 0:
                # select candidate_trees
                classifier.select_candidate_trees(transition_tree_pos_list)
            if len(adapt_state_tree_pos_list) > 0:
                # update actual drifted trees

                actual_drifted_tree_indices = \
                    classifier.adapt_state(adapt_state_tree_pos_list, False)
                print(f"LOG: actual_drifted_tree_indices: {actual_drifted_tree_indices}")

            # log performance
            if count % sample_freq == 0 and count != 0:
                elapsed_time = time.process_time() - start_time + cpt_runtime
                accuracy = correct / sample_freq
                kappa = cohen_kappa_score(window_actual_labels, window_predicted_labels)
                candidate_tree_size = classifier.get_candidate_tree_group_size()
                tree_pool_size = classifier.get_tree_pool_size()

                print(f"{count},{accuracy},{kappa},{candidate_tree_size},{tree_pool_size},{str(elapsed_time)}")

                correct = 0
                window_actual_labels = []
                window_predicted_labels = []

    # TODO
    assert True
