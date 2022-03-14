# import math
# import time
from skika.transfer import opera_wrapper

expected_accuracies = \
    [[0.09, 0.11, 0.16, 0.24, 0.3 , 0.29, 0.29, 0.32, 0.36, 0.34], \
    [0.27, 0.27, 0.22, 0.21, 0.3, 0.42, 0.47, 0.49, 0.47, 0.49]]

expected_full_region_complexity = 9
expected_error_region_complexity = 8
expected_correct_region_complexity = 5

def test_arf():

    max_samples = 10000
    sample_freq = 1000
    num_trees = 100
    rf_lambda = 1
    random_state = 0

    num_phantom_branches=30
    squashing_delta=7
    obs_period=1000
    conv_delta=0.1
    conv_threshold=0.15
    obs_window_size=50
    perf_window_size=5000
    min_obs_period=2000
    split_range=10
    force_disable_patching=False
    force_enable_patching=False
    grow_transfer_surrogate_during_obs=False


    data_file_path="./transfer-data/fashion-mnist/flip20/source.arff;./transfer-data/fashion-mnist/flip20/target.arff";

    classifier = opera_wrapper(
        len(data_file_path.split(";")),
        random_state,
        num_trees,
        rf_lambda,
        num_phantom_branches,
        squashing_delta,
        obs_period,
        conv_delta,
        conv_threshold,
        obs_window_size,
        perf_window_size,
        min_obs_period,
        split_range,
        grow_transfer_surrogate_during_obs,
        force_disable_patching,
        force_enable_patching)

    data_file_list = data_file_path.split(";")


    prequential_evaluation_transfer(
        classifier=classifier,
        data_file_paths=data_file_list,
        max_samples=max_samples,
        sample_freq=sample_freq,
        expected_accuracies=expected_accuracies)

    # for count in range(0, max_samples):
    #     if not classifier.get_next_instance():
    #         break

    #     # test
    #     prediction = classifier.predict()

    #     actual_label = classifier.get_cur_instance_label()
    #     if prediction == actual_label:
    #         correct += 1

    #     if count % sample_freq == 0 and count != 0:
    #         accuracy = correct / sample_freq
    #         assert accuracy == expected_accuracies[int(count/sample_freq) - 1]
    #         correct = 0

    #     # train
    #     classifier.train()

    #     classifier.delete_cur_instance()

class ClassifierMetrics:
    def __init__(self):
        self.correct = 0
        self.instance_idx = 0

def prequential_evaluation_transfer(
        classifier,
        data_file_paths,
        max_samples,
        sample_freq,
        expected_accuracies):

    classifier_metrics_list = []
    for i in range(len(data_file_paths)):
        classifier.init_data_source(i, data_file_paths[i])
        classifier_metrics_list.append(ClassifierMetrics())

    classifier_idx = 0
    classifier.switch_classifier(classifier_idx)
    metric = classifier_metrics_list[classifier_idx]

    while True:
        if not classifier.get_next_instance() \
                or classifier_metrics_list[classifier_idx].instance_idx == max_samples:
            # Switch streams to simulate parallel streams

            classifier_idx += 1
            if classifier_idx >= len(data_file_paths):
                break

            classifier.switch_classifier(classifier_idx)
            metric = classifier_metrics_list[classifier_idx]

            print()
            print(f"switching to classifier_idx {classifier_idx}")
            continue

        classifier_metrics_list[classifier_idx].instance_idx += 1

        # test
        prediction = classifier.predict()

        actual_label = classifier.get_cur_instance_label()
        if prediction == actual_label:
            metric.correct += 1

        # train
        classifier.train()

        log_metrics(
            classifier_metrics_list[classifier_idx].instance_idx,
            sample_freq,
            metric,
            classifier_idx,
            classifier)

def log_metrics(count, sample_freq, metric, classifier_idx, classifier):
    if count % sample_freq == 0 and count != 0:
        accuracy = round(metric.correct / sample_freq, 2)
        assert accuracy == expected_accuracies[classifier_idx][int(count/sample_freq) - 1]

        f = int(classifier.get_full_region_complexity())
        e = int(classifier.get_error_region_complexity())
        c = int(classifier.get_correct_region_complexity())

        if classifier_idx > 0:
            assert f == expected_full_region_complexity
            assert e == expected_error_region_complexity
            assert c == expected_correct_region_complexity

        # print(f"{count},{accuracy},{f},{e},{c}")
        metric.correct = 0
