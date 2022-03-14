#include "opera_wrapper.h"

opera_wrapper::opera_wrapper(
        int num_classifiers,
        int seed,
        int num_trees,
        int lambda,
        // transfer learning params
        int num_phantom_branches,
        int squashing_delta,
        int obs_period,
        double conv_delta,
        double conv_threshold,
        int obs_window_size,
        int perf_window_size,
        int min_obs_period,
        int split_range,
        bool grow_transfer_surrogate_during_obs,
        bool force_disable_patching,
        bool force_enable_patching) {

    for (int i = 0; i < num_classifiers; i++) {
        shared_ptr<opera> classifier = make_shared<opera>(
            seed,
            num_trees,
            lambda,
            // transfer learning params
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
            force_enable_patching);

        classifiers.push_back(classifier);
    }

    for (int i = 0; i < num_classifiers; i++) {
        for (int j = 0; j < num_classifiers; j++) {
            if (i == j) continue;
            classifiers[i]->register_tree_pool(classifiers[j]->get_concept_repo());
        }
    }
}

void opera_wrapper::switch_classifier(int classifier_idx) {
    current_classifier = classifiers[classifier_idx];
}

void opera_wrapper::train() {
    current_classifier->train();
}

int opera_wrapper::predict() {
    return current_classifier->predict();
}

int opera_wrapper::get_cur_instance_label() {
    return current_classifier->get_cur_instance_label();
}

void opera_wrapper::init_data_source(int classifier_idx, const string &filename) {
    classifiers[classifier_idx]->init_data_source(filename);
}

bool opera_wrapper::get_next_instance() {
    return current_classifier->get_next_instance();
}

double opera_wrapper::get_full_region_complexity() {
    return current_classifier->get_full_region_complexity();
}

double opera_wrapper::get_error_region_complexity() {
    return current_classifier->get_error_region_complexity();
}

double opera_wrapper::get_correct_region_complexity() {
    return current_classifier->get_correct_region_complexity();
}
