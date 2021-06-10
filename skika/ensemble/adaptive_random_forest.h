#ifndef __ADAPTIVE_RANDOM_FOREST_H__
#define __ADAPTIVE_RANDOM_FOREST_H__

#include <memory>
#include <string>
#include <climits>
#include <random>

#include <streamDM/streams/ArffReader.h>
#include <streamDM/learners/Classifiers/Trees/HoeffdingTree.h>
#include <streamDM/learners/Classifiers/Trees/ADWIN.h>

#define LOG(x) std::cout << (x) << std::endl

using std::string;
using std::unique_ptr;
using std::shared_ptr;
using std::vector;
using std::make_unique;
using std::make_shared;
using std::move;

class arf_tree;

class adaptive_random_forest {

    public:

        adaptive_random_forest(int num_trees,
                               int arf_max_features,
                               int lambda,
                               int seed,
                               double warning_delta,
                               double drift_delta);

        std::mt19937 mrand;

        bool init_data_source(const string& filename);
        bool get_next_instance();
        int get_cur_instance_label();
        void delete_cur_instance();

        virtual int predict();
        virtual void train();
        int vote(const vector<int>& votes);

    protected:

        int num_trees;
        int num_features;
        int arf_max_features;
        int lambda;
        double warning_delta;
        double drift_delta;

        Instance* instance;
        unique_ptr<Reader> reader;

        vector<shared_ptr<arf_tree>> foreground_trees;

        virtual void init();
        shared_ptr<arf_tree> make_arf_tree();
        bool detect_change(int error_count, unique_ptr<HT::ADWIN>& detector);
};

class arf_tree {
    public:
        arf_tree(double warning_delta,
                 double drift_delta,
                 std::mt19937 mrand);

        virtual void train(Instance& instance);
        virtual int predict(Instance& instance);

        unique_ptr<HT::HoeffdingTree> tree;
        shared_ptr<arf_tree> bg_arf_tree;
        unique_ptr<HT::ADWIN> warning_detector;
        unique_ptr<HT::ADWIN> drift_detector;

    protected:
        double warning_delta;
        double drift_delta;
        std::mt19937 mrand;
};

#endif
