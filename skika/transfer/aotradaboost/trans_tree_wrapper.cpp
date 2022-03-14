#include "trans_tree_wrapper.h"

trans_tree_wrapper::trans_tree_wrapper(
        int num_classifiers,
        int seed,
        int kappa_window_size,
        double warning_delta,
        double drift_delta,
        // transfer learning params
        int least_transfer_warning_period_instances_length,
        int instance_store_size,
        int num_diff_distr_instances,
        int bbt_pool_size,
        int eviction_interval,
        double transfer_kappa_threshold,
        double gamma,
        double transfer_match_lowerbound,
        string boost_mode_str,
        int num_trees,
        bool disable_drift_detection) {

    for (int i = 0; i < num_classifiers; i++) {
        shared_ptr<trans_tree> classifier = make_shared<trans_tree>(
                seed,
                kappa_window_size,
                warning_delta,
                drift_delta,
                least_transfer_warning_period_instances_length,
                instance_store_size,
                num_diff_distr_instances,
                bbt_pool_size,
                eviction_interval,
                transfer_kappa_threshold,
                gamma,
                transfer_match_lowerbound,
                boost_mode_str,
                num_trees,
                disable_drift_detection);

        classifiers.push_back(classifier);
    }

    for (int i = 0; i < num_classifiers; i++) {
        for (int j = 0; j < num_classifiers; j++) {
            if (i == j) continue;
            classifiers[i]->register_tree_pool(classifiers[j]->get_concept_repo());
        }
    }
}

void trans_tree_wrapper::switch_classifier(int classifier_idx){
    current_classifier = classifiers[classifier_idx];
}

void trans_tree_wrapper::train() {
    current_classifier->train();
}

int trans_tree_wrapper::predict() {
    return current_classifier->predict();
}

int trans_tree_wrapper::get_cur_instance_label() {
    return current_classifier->get_cur_instance_label();
}

void trans_tree_wrapper::init_data_source(int classifier_idx, const string &filename){
    classifiers[classifier_idx]->init_data_source(filename);
}

bool trans_tree_wrapper::get_next_instance() {
    return current_classifier->get_next_instance();
}

int trans_tree_wrapper::get_transferred_tree_group_size() {
    return current_classifier->get_transferred_tree_group_size();
}

int trans_tree_wrapper::get_tree_pool_size() {
    return current_classifier->get_tree_pool_size();
}
