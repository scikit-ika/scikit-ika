#ifndef __PRO_PEARL_H__
#define __PRO_PEARL_H__

#include "pearl.h"

class pro_pearl : public pearl {

    public:

        pro_pearl(int num_trees,
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
                  int pro_drift_window_size,
                  double hybrid_delta,
                  int backtrack_window,
                  double stability_delta);

        virtual void train();
        virtual shared_ptr<pearl_tree> make_pearl_tree(int tree_pool_id);
        virtual void init();

        int find_last_actual_drift_point(int tree_idx);
        void set_expected_drift_prob(int tree_idx, double p);
        bool has_actual_drift(int tree_idx);
        vector<int> get_stable_tree_indices();

        void select_predicted_trees(const vector<int>& warning_tree_pos_list);

        vector<int> adapt_state(const vector<int>& drifted_tree_pos_list, bool is_proactive);
        vector<int> adapt_state_with_proactivity(
                const vector<int>& drifted_tree_pos_list,
                deque<shared_ptr<pearl_tree>>& _candidate_trees);

    private:

        int pro_drift_window_size = 100;
        double hybrid_delta = 0.001;
        int backtrack_window = 25;
        double stability_delta = 0.001;

        int num_max_backtrack_instances = 100000000; // TODO
        int num_instances_seen = 0;
        deque<Instance*> backtrack_instances;
        set<int> potential_drifted_tree_indices;
        vector<unique_ptr<HT::ADWIN>> stability_detectors;
        vector<int> stable_tree_indices;
        deque<shared_ptr<pearl_tree>> predicted_trees;

        static bool compare_kappa_arf(shared_ptr<arf_tree>& tree1,
                                      shared_ptr<arf_tree>& tree2);
        // virtual void predict_with_state_adaption(vector<int>& votes, int actual_label);
        bool detect_stability(int error_count, unique_ptr<HT::ADWIN>& detector);
};

#endif
