from skika.ensemble import adaptive_random_forest

def test_arf():
    expected_accuracies = \
        [0.583, 0.357, 0.298, 0.532, 0.535, 0.367, 0.35, 0.358, 0.393, 0.564, 0.492, 0.459,
         0.379, 0.267, 0.416, 0.763, 0.792, 0.784, 0.745, 0.799, 0.807, 0.843, 0.856, 0.769,
         0.804, 0.818, 0.851, 0.783, 0.815, 0.82, 0.805, 0.782, 0.755, 0.742, 0.774, 0.684,
         0.761, 0.777, 0.819, 0.792,]

    num_trees = 60
    max_features = -1
    warning_delta = 0.0001
    drift_delta = 0.00001

    max_samples = 5000

    classifier = adaptive_random_forest(num_trees,
                                        max_features,
                                        warning_delta,
                                        drift_delta)
    classifier.init_data_source("./recurrent-data/real-world/covtype.arff");

    correct = 0
    max_samples = 40001
    sample_freq = 1000

    for count in range(0, max_samples):
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
