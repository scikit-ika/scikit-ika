#ifndef TRANS_TREE_WRAPPER_H
#define TRANS_TREE_WRAPPER_H

#include "trans_tree.h"

class trans_tree_wrapper {

public:
    trans_tree_wrapper(
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
            bool disable_drift_detection);

    void switch_classifier(int classifier_idx);
    void train();
    int predict();
    int get_cur_instance_label();
    void init_data_source(int classifier_idx, const string &filename);
    bool get_next_instance();
    int get_transferred_tree_group_size();
    int get_tree_pool_size();

private:

    vector<shared_ptr<trans_tree>> classifiers;
    shared_ptr<trans_tree> current_classifier;

};

#endif //TRANS_TREE_WRAPPER_H
