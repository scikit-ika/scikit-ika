#ifndef TRANS_TREE_H
#define TRANS_TREE_H

#include <random>
#include <streamDM/streams/ArffReader.h>
#include <streamDM/learners/Classifiers/Trees/HoeffdingTree.h>

#include "random_forest.h"
#include "phantom_tree.h"

class opera {

    class instance_store_complexity;
    class true_error;

public:
    opera(
            int seed,
            int num_trees,
            double lambda,
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

    void train();
    int predict();
    void init();
    shared_ptr<random_forest> make_tree();
    bool init_data_source(const string &filename);
    bool get_next_instance();
    int get_cur_instance_label();
    void delete_cur_instance();

    bool transfer_model();
    unique_ptr<phantom_tree> make_phantom_tree();
    bool is_high_adaptability();

    double get_full_region_complexity();
    double get_error_region_complexity();
    double get_correct_region_complexity();

    bool force_disable_patching = false;
    bool force_enable_patching = false;

    // transfer
    vector<shared_ptr<random_forest>> &get_concept_repo();
    void register_tree_pool(vector<shared_ptr<random_forest>> &pool);
    // TODO simplify
    vector<vector<shared_ptr<random_forest>>*> registered_tree_pools;

private:

    // int kappa_window_size;
    std::mt19937 mrand;
    shared_ptr<random_forest> classifier = nullptr;
    vector<shared_ptr<random_forest>> tree_pool;
    deque<int> actual_labels;

    Instance* instance;
    unique_ptr<Reader> reader;

    ////////////////////////////////////////////////////////

    int seed;
    int num_trees;
    double lambda;
    // transfer learning params
    int num_phantom_branches;
    int squashing_delta;
    int obs_period;
    double conv_delta;
    double conv_threshold;
    int obs_window_size;
    int perf_window_size;
    int min_obs_period;
    int split_range;
    bool grow_transfer_surrogate_during_obs;

    ////////////////////////////////////////////////////////

    shared_ptr<random_forest> transferSurrogate = nullptr;

    vector<Instance*> obsInstanceStore;
    vector<int> obsPredictionResults;
    vector<Instance*> errorRegionInstanceStore;
    vector<Instance*> aproposRegionInstanceStore;
    unique_ptr<true_error> m_true_error = nullptr;

    shared_ptr<random_forest> errorRegionClassifier = nullptr;
    shared_ptr<random_forest> patchClassifier = nullptr;
    shared_ptr<random_forest> newClassifier = nullptr;

    bool inObsPeriod = false;

    // track performances for both the patch learner and the transferred model
    deque<int> patchErrorWindow;
    deque<int> transErrorWindow;
    deque<int> newErrorWindow;
    double patchErrorWindowSum = 0.0;
    double transErrorWindowSum = 0.0;
    double newErrorWindowSum = 0.0;

    double full_region_complexity = -1;
    double error_region_complexity = -1;
    double correct_region_complexity = -1;

    bool switch_to_new_classifier();
    bool turn_on_patch_prediction();

    class true_error {
        public:
            true_error(
                    int windowSize,
                    double delta,
                    double convThreshold,
                    std::mt19937& mrand);

            bool is_stable(int error);

        private:
            int sampleSize = 0;
            int windowSize = 0;
            double rc;
            double errorCount;
            double delta;
            double convThreshold;
            double windowSum = 0.0;
            deque<double> window;
            std::mt19937 mrand;

        double get_true_error(int error);
    };

};

#endif //TRANS_TREE_H