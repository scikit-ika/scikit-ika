from skika.ensemble import pearl

def test_pearl():
    expected_accuracies = \
        [0.583, 0.357, 0.298, 0.532, 0.535, 0.367, 0.35, 0.358, 0.393, 0.564, 0.492,
         0.464, 0.384, 0.274, 0.427, 0.763, 0.792, 0.784, 0.745, 0.799, 0.807, 0.843,
         0.856, 0.769, 0.804, 0.818, 0.851, 0.783, 0.815, 0.82, 0.81, 0.786, 0.755, 0.739,
         0.795, 0.694, 0.769, 0.794, 0.846, 0.939]

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

    classifier = pearl(num_trees,
                       max_num_candidate_trees,
                       repo_size,
                       edit_distance_threshold,
                       kappa_window,
                       lossy_window_size,
                       reuse_window_size,
                       max_features,
                       bg_kappa_threshold,
                       cd_kappa_threshold,
                       reuse_rate_upper_bound,
                       warning_delta,
                       drift_delta,
                       enable_state_adaption,
                       enable_state_graph)
    classifier.init_data_source("recurrent-data/real-world/covtype.arff");

    correct = 0
    max_samples = 40001
    sample_freq = 1000

    for count in range(max_samples):
        if not classifier.get_next_instance():
            break

        # test
        prediction = classifier.predict()

        actual_label = classifier.get_cur_instance_label()
        if prediction == actual_label:
            correct += 1

        if count % sample_freq == 0 and count != 0:
            accuracy = correct / sample_freq
            assert accuracy == expected_accuracies[int(count/sample_freq) - 1]
            correct = 0

        # train
        classifier.train()

        classifier.delete_cur_instance()
