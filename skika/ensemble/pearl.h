#ifndef __PEARL_H__
#define __PEARL_H__

#include <deque>
#include "adaptive_random_forest.h"
#include "lru_state.h"
#include "lossy_state_graph.h"

using std::set;

class pearl_tree;

class pearl : public adaptive_random_forest {

    public:

        pearl(int num_trees,
              int max_num_candidate_trees,
              int repo_size,
              int edit_distance_threshold,
              int kappa_window_size,
              int lossy_window_size,
              int reuse_window_size,
              int arf_max_features,
              int lambda,
              int seed,
              double bg_kappa_threshold,
              double cd_kappa_threshold,
              double reuse_rate_upper_bound,
              double warning_delta,
              double drift_delta,
              bool enable_state_adaption,
              bool enable_state_graph);

        virtual void train();
        int get_candidate_tree_group_size() const;
        int get_tree_pool_size() const;

        void select_candidate_trees(const vector<int>& warning_tree_pos_list);
        void tree_transition(const vector<int>& warning_tree_pos_list);
        void tree_transition(
                const vector<int>& warning_tree_pos_list,
                deque<shared_ptr<pearl_tree>>& _candidate_trees);
        void pattern_match_candidate_trees(const vector<int>& warning_tree_pos_list);
        void pattern_match_candidate_trees(
                const vector<int>& warning_tree_pos_list,
                deque<shared_ptr<pearl_tree>>& _candidate_trees);

        static bool compare_kappa(shared_ptr<pearl_tree>& tree1,
                                  shared_ptr<pearl_tree>& tree2);

        bool is_state_graph_stable();


    protected:

        int max_num_candidate_trees;
        int repo_size;
        int edit_distance_threshold;
        int kappa_window_size;
        int lossy_window_size;
        int reuse_window_size;
        double bg_kappa_threshold;
        double cd_kappa_threshold;
        double reuse_rate_upper_bound;
        bool enable_state_adaption;
        bool enable_state_graph;

        deque<shared_ptr<pearl_tree>> candidate_trees;
        vector<shared_ptr<pearl_tree>> tree_pool;

        unique_ptr<state_graph_switch> graph_switch;
        shared_ptr<lossy_state_graph> state_graph;
        unique_ptr<lru_state> state_queue;
        set<int> cur_state;
        deque<int> actual_labels;

        virtual void adapt_state(const vector<int>& drifted_tree_pos_list);
        virtual shared_ptr<pearl_tree> make_pearl_tree(int tree_pool_id);
        virtual void init();
};

class pearl_tree : public arf_tree {
    public:
        int tree_pool_id;
        double kappa = INT_MIN;
        bool is_candidate = false;
        deque<int> predicted_result_right_window;
        deque<int> predicted_result_left_window;
        deque<int> predicted_labels_window; // of size kappa_window_size

        pearl_tree(int tree_pool_id,
                   int kappa_window_size,
                   double warning_delta,
                   double drift_delta,
                   std::mt19937 mrand);

        pearl_tree(int tree_pool_id,
                   int kappa_window_size,
                   int pro_drift_window_size,
                   double warning_delta,
                   double drift_delta,
                   double hybrid_delta,
                   std::mt19937 mrand);

        virtual void train(Instance& instance);
        virtual int predict(Instance& instance, bool track_performance);
        virtual void reset();
        void update_kappa(const deque<int>& actual_labels, int class_count);
        void set_expected_drift_prob(double p);
        bool has_actual_drift();

        shared_ptr<pearl_tree> bg_pearl_tree = nullptr;
        shared_ptr<pearl_tree> replaced_tree = nullptr;

    private:
        int kappa_window_size;
        int pro_drift_window_size = 0;
        double hybrid_delta = 0.001;

        double left_correct_count = 0.0;
        double right_correct_count = 0.0;

        double get_variance();
        double compute_hoeffding_bound(double variance, double window_size, double delta);

        double compute_kappa(const vector<vector<int>>& confusion_matrix,
                             double accuracy,
                             int sample_count,
                             int class_count);
};

#endif
