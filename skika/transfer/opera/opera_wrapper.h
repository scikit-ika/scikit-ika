#ifndef TRANS_TREE_WRAPPER_H
#define TRANS_TREE_WRAPPER_H

#include "opera.h"
#include "random_forest.h"

class opera_wrapper {

public:
    opera_wrapper(
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
            bool force_enable_patching);

    void switch_classifier(int classifier_idx);
    void train();
    int predict();
    int get_cur_instance_label();
    void init_data_source(int classifier_idx, const string &filename);
    bool get_next_instance();
    int get_transferred_tree_group_size();
    int get_tree_pool_size();

    double get_full_region_complexity();
    double get_error_region_complexity();
    double get_correct_region_complexity();

private:

    vector<shared_ptr<opera>> classifiers;
    shared_ptr<opera> current_classifier;

};

#endif //TRANS_TREE_WRAPPER_H