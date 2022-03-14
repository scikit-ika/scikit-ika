from skika.aotradaboost import trans_tree_wrapper

def test_arf():
    expected_accuracies = \
        []

    sample_freq = 1000
    num_trees = 10 # 60
    rf_lambda = 1
    random_state = 0

    warning_delta = 0.0001
    drift_delta = 0.00001
    kappa_window = 60

    least_transfer_warning_period_instances_length = 300
    instance_store_size = 8000
    num_diff_distr_instances = 200
    bbt_pool_size = 40
    eviction_interval = 1000000
    transfer_kappa_threshold = 0.1
    transfer_gamma = 8
    transfer_match_lowerbound = 0.0
    boost_mode = "atradaboost" # i.e. aotradaboost
    disable_drift_detection = False


    data_file_path = "./transfer-data/bike/dc-weekend-source.arff;./transfer-data/bike/weekday.arff";

    classifier = trans_tree_wrapper(
        len(data_file_path.split(";")),
        random_state,
        kappa_window,
        warning_delta,
        drift_delta,
        least_transfer_warning_period_instances_length,
        instance_store_size,
        num_diff_distr_instances,
        bbt_pool_size,
        eviction_interval,
        transfer_kappa_threshold,
        transfer_gamma,
        transfer_match_lowerbound,
        boost_mode,
        num_trees,
        disable_drift_detection)

    data_file_list = data_file_path.split(";")


    prequential_evaluation_transfer(
        classifier=classifier,
        data_file_paths=data_file_list,
        sample_freq=sample_freq,
        expected_accuracies=expected_accuracies)

class ClassifierMetrics:
    def __init__(self):
        self.correct = 0
        self.instance_idx = 0

def prequential_evaluation_transfer(
        classifier,
        data_file_paths,
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
        if not classifier.get_next_instance():
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
            classifier)

def log_metrics(count, sample_freq, metric, classifier):
    if count % sample_freq == 0 and count != 0:
        accuracy = round(metric.correct / sample_freq, 2)
        print(f"{accuracy}")

        metric.correct = 0
