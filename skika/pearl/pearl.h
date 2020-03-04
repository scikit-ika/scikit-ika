#ifndef __PEARL_H__
#define __PEARL_H__

#include <deque>
#include <memory>
#include <string>
#include <climits>
#include <random>

#include <streamDM/streams/ArffReader.h>
#include <streamDM/learners/Classifiers/Trees/HoeffdingTree.h>
#include <streamDM/learners/Classifiers/Trees/ADWIN.h>

#include "lru_state.h"
#include "lossy_state_graph.h"

#define LOG(x) std::cout << (x) << std::endl

using std::string;
using std::unique_ptr;
using std::shared_ptr;
using std::make_unique;
using std::make_shared;
using std::move;
using std::vector;
using std::set;

class pearl {

    public:

        std::mt19937 mrand;

        class adaptive_tree {
            public:
                int tree_pool_id;
                double kappa = INT_MIN;
                bool is_candidate = false;
                deque<int> predicted_labels;

                adaptive_tree(int tree_pool_id,
                              int kappa_window_size,
                              double warning_delta,
                              double drift_delta);

                void train(Instance& instance);
                int predict(Instance& instance, bool track_performance);
                void update_kappa(const deque<int>& actual_labels, int class_count);
                void reset();

                unique_ptr<HT::HoeffdingTree> tree;
                shared_ptr<adaptive_tree> bg_adaptive_tree;
                unique_ptr<HT::ADWIN> warning_detector;
                unique_ptr<HT::ADWIN> drift_detector;

            private:
                int kappa_window_size;
                double warning_delta;
                double drift_delta;

                double compute_kappa(const vector<vector<int>>& confusion_matrix,
                                     double accuracy,
                                     int sapmle_count,
                                     int class_count);
        };

        pearl(int num_trees,
              int max_num_candidate_trees,
              int repo_size,
              int edit_distance_threshold,
              int kappa_window_size,
              int lossy_window_size,
              int reuse_window_size,
              int arf_max_features,
              double bg_kappa_threshold,
              double cd_kappa_threshold,
              double reuse_rate_upper_bound,
              double warning_delta,
              double drift_delta,
              bool enable_state_adaption,
              bool enable_state_graph);

        void init();

        int get_candidate_tree_group_size() const;
        int get_tree_pool_size() const;

        bool init_data_source(const string& filename);
        bool get_next_instance();
        int get_cur_instance_label();
        void delete_cur_instance();
        void prepare_instance(Instance& instance);

        virtual int predict();
        virtual void train();
        int vote(const vector<int>& votes);

        void select_candidate_trees(const vector<int>& warning_tree_pos_list);
        void tree_transition(const vector<int>& warning_tree_pos_list);
        void pattern_match_candidate_trees(const vector<int>& warning_tree_pos_list);

        static bool compare_kappa(shared_ptr<adaptive_tree>& tree1,
                                  shared_ptr<adaptive_tree>& tree2);

        bool is_state_graph_stable();


    protected:

        int num_trees;
        int max_num_candidate_trees;
        int repo_size;
        int edit_distance_threshold;
        int kappa_window_size;
        int lossy_window_size;
        int reuse_window_size;
        int num_features;
        int arf_max_features;
        double bg_kappa_threshold;
        double cd_kappa_threshold;
        double reuse_rate_upper_bound;
        double warning_delta;
        double drift_delta;
        bool enable_state_adaption;
        bool enable_state_graph;

        Instance* instance;
        unique_ptr<Reader> reader;

        vector<shared_ptr<adaptive_tree>> adaptive_trees;
        deque<shared_ptr<adaptive_tree>> candidate_trees;
        vector<shared_ptr<adaptive_tree>> tree_pool;

        unique_ptr<state_graph_switch> graph_switch;
        shared_ptr<lossy_state_graph> state_graph;
        unique_ptr<lru_state> state_queue;
        set<int> cur_state;
        deque<int> actual_labels;

        bool detect_change(int error_count, unique_ptr<HT::ADWIN>& detector);
        shared_ptr<adaptive_tree> make_adaptive_tree(int tree_pool_id);
        void online_bagging(Instance& instance, adaptive_tree& tree);

        void predict_basic(vector<int>& votes, int actual_label);
        void predict_with_state_adaption(vector<int>& votes, int actual_label);
        virtual void adapt_state(const vector<int>& drifted_tree_pos_list);

};

#endif
