#ifndef __ADAPTIVE_RANDOM_FOREST_H__
#define __ADAPTIVE_RANDOM_FOREST_H__

#include <memory>
#include <string>
#include <climits>
#include <random>

#include <streamDM/streams/ArffReader.h>
#include <streamDM/learners/Classifiers/Trees/HoeffdingTree.h>
#include <streamDM/learners/Classifiers/Trees/ADWIN.h>

using std::string;
using std::unique_ptr;
using std::shared_ptr;
using std::vector;
using std::make_unique;
using std::make_shared;
using std::move;

struct tree_params_t {
    int grace_period = 50;
    float split_confidence = 0.01; // 0.0000001f;
    float tie_threshold = 0.1; // 0.05;
    bool binary_splits = false;
    bool no_pre_prune = false;
    int nb_threshold = 0;
    int leaf_prediction_type = 0;
};

class rf_tree;

class random_forest {

    public:
        // random_forest(
        //         int num_trees,
        //         double lambda,
        //         int seed,
        //         int grace_period,
        //         float split_confidence,
        //         float tie_threshold,
        //         bool binary_splits,
        //         bool no_pre_prune,
        //         int nb_threshold,
        //         int leaf_prediction_type);

        random_forest(
                int num_trees,
                double lambda,
                std::mt19937& mrand);

        std::mt19937 mrand;

        int get_cur_instance_label();
        void delete_cur_instance();

        virtual int predict(Instance* instance);
        virtual void train(Instance* instance);
        int vote(const vector<int>& votes);

protected:

        int num_trees;
        double lambda;
        vector<shared_ptr<rf_tree>> foreground_trees;

        virtual void init();
        shared_ptr<rf_tree> make_rf_tree();

        tree_params_t tree_params;
};

class rf_tree {
    public:
        rf_tree(tree_params_t tree_params,
                 std::mt19937& mrand);

        virtual void train(Instance& instance);
        virtual int predict(Instance& instance);

        unique_ptr<HT::HoeffdingTree> tree;
};

#endif
