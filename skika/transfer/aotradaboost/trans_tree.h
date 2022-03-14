#ifndef TRANS_TREE_H
#define TRANS_TREE_H

#include <random>
#include <streamDM/streams/ArffReader.h>
#include <streamDM/learners/Classifiers/Trees/HoeffdingTree.h>
#include <streamDM/learners/Classifiers/Trees/ADWIN.h>

enum class boost_modes_enum { no_boost_mode, ozaboost_mode, tradaboost_mode, otradaboost_mode, atradaboost_mode };
double compute_kappa(deque<int> predicted_labels, deque<int> actual_labels, int class_count);

class hoeffding_tree;
class trans_tree {
    class boosted_bg_tree_pool;

public:
    trans_tree(
            int seed,
            int kappa_window_size,
            double warning_delta,
            double drift_delta,
            // transfer learning params
            int least_transfer_warning_period_instances_length, // tuning required
            int instance_store_size,
            int num_diff_distr_instances,
            int bbt_pool_size, // tuning required
            int eviction_interval,
            double transfer_kappa_threshold,
            double gamma,
            double transfer_match_lowerbound,
            string boost_mode_str,
            int num_trees,
            bool disable_drift_detection);

    void train();
    int predict();
    void init();
    shared_ptr<hoeffding_tree> make_tree(int tree_pool_id);
    static bool detect_change(int error_count, unique_ptr<HT::ADWIN>& detector);

    int get_transferred_tree_group_size();
    int get_tree_pool_size();

    bool init_data_source(const string& filename);
    bool get_next_instance();
    int get_cur_instance_label();
    void delete_cur_instance();

    // transfer
    vector<shared_ptr<hoeffding_tree>>& get_concept_repo();
    void register_tree_pool(vector<shared_ptr<hoeffding_tree>>& pool);
    bool transfer(Instance* instance);
    shared_ptr<hoeffding_tree> match_concept(vector<Instance*> warning_period_instances);
    int get_transferred_tree_group_size() const;
    int transferred_tree_total_count = 0;
    // double compute_kappa(vector<int> predicted_labels, vector<int> actual_labels, int class_count);
    vector<vector<shared_ptr<hoeffding_tree>>*> registered_tree_pools;

private:
    bool enable_transfer = true;
    double gamma = 3.0;
    double transfer_match_lowerbound = 0.0;

    int kappa_window_size;
    double warning_delta;
    double drift_delta;
    std::mt19937 mrand;
    shared_ptr<hoeffding_tree> foreground_tree;
    vector<shared_ptr<hoeffding_tree>> tree_pool;
    deque<int> actual_labels;

    Instance* instance;
    unique_ptr<Reader> reader;

    // transfer
    std::map<string, boost_modes_enum> boost_mode_map =
            {
                    { "no_boost", boost_modes_enum::no_boost_mode},
                    { "ozaboost", boost_modes_enum::ozaboost_mode },
                    { "tradaboost", boost_modes_enum::tradaboost_mode },
                    { "otradaboost", boost_modes_enum::otradaboost_mode },
                    { "atradaboost", boost_modes_enum::atradaboost_mode },
            };
    boost_modes_enum boost_mode = boost_modes_enum::otradaboost_mode;
    int least_transfer_warning_period_length = 50;
    int instance_store_size = 500;
    int num_diff_distr_instances = 30;
    int bbt_pool_size = 100;
    int eviction_interval = 100;
    int num_trees = 1;
    bool disable_drift_detection = false;
    double transfer_kappa_threshold = 0.3;
    unique_ptr<boosted_bg_tree_pool> bbt_pool;

    class boosted_bg_tree_pool {
    public:
        boost_modes_enum boost_mode = boost_modes_enum::otradaboost_mode;
        double weight_factor = 1.0;

        boosted_bg_tree_pool(enum boost_modes_enum boost_mode,
                             int pool_size,
                             int eviction_interval,
                             double transfer_kappa_threshold,
                             shared_ptr<hoeffding_tree> tree_template,
                             int lambda);

        // training starts when a mini_batch is ready
        void train(Instance* instance, bool is_same_distribution);
        shared_ptr<hoeffding_tree> get_best_model(deque<int> actual_labels, int class_count);
        void online_boost(Instance* instance, bool _is_same_distribution);
        Instance* get_next_diff_distr_instance();

        vector<Instance*> warning_period_instances;
        shared_ptr<hoeffding_tree> matched_tree = nullptr;
        int instance_store_idx = 0;

        vector<double> oob_tree_correct_lam_sum; // count of out-of-bag correctly predicted trees per instance
        vector<double> oob_tree_wrong_lam_sum; // count of out-of-bag incorrectly predicted trees per instance
        vector<double> oob_tree_lam_sum; // count of oob trees per instance


        // online tradaboost
        vector<double> lam_sum_correct_src;
        vector<double> lam_sum_wrong_src;
        vector<double> error_src;

        vector<double> lam_sum_correct_tgt;
        vector<double> lam_sum_wrong_tgt;
        vector<double> error_tgt;
        vector<double> weight_distri_tgt;

        double num_src_instances;


    private:
        double lambda = 1;
        double epsilon = 1;
        std::mt19937 mrand;

        long pool_size = 10;
        long bbt_counter = 0;
        long boost_count = 0;
        long eviction_interval = 100;
        double transfer_kappa_threshold = 0.3;
        shared_ptr<hoeffding_tree> tree_template;
        vector<shared_ptr<hoeffding_tree>> pool;

        // execute replacement strategies when the bbt pool is full
        void update_bbt();
        void no_boost(Instance* instance);
        void ozaboost(Instance* instance);
        void tradaboost(Instance* instance, bool is_same_distribution);
        void otradaboost(Instance* instance, bool is_same_distribution);
        void atradaboost(Instance* instance, bool is_same_distribution);
        void perf_eval(Instance* instance);
    };
};

class hoeffding_tree {
public:
    hoeffding_tree(double warning_delta, double drift_delta, int instance_store_size, int num_trees, std::mt19937 mrand);
    hoeffding_tree(hoeffding_tree const &rhs);

    void train(Instance& instance);
    int predict(Instance& instance, bool track_prediction);
    int vote(const vector<int>& votes);
    void store_instance(Instance* instance);

    // unique_ptr<HT::HoeffdingTree> tree;
    vector<shared_ptr<HT::HoeffdingTree>> tree;
    shared_ptr<hoeffding_tree> bg_tree;
    unique_ptr<HT::ADWIN> warning_detector;
    unique_ptr<HT::ADWIN> drift_detector;
    int tree_pool_id = -1;
    double kappa = numeric_limits<double>::min();
    deque<int> predicted_labels;
    int kappa_window_size = 60;
    int num_trees;
    std::poisson_distribution<int> poisson_distr;

    deque<Instance*> instance_store;
    int instance_store_size;
    double warning_period_kappa = std::numeric_limits<double>::min();

private:
    double warning_delta;
    double drift_delta;
    std::mt19937 mrand;
};

#endif //TRANS_TREE_H
