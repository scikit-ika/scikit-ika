#include "trans_tree.h"

trans_tree::trans_tree(
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
        bool disable_drift_detection) :
    kappa_window_size(kappa_window_size),
    warning_delta(warning_delta),
    drift_delta(drift_delta),
    least_transfer_warning_period_length(least_transfer_warning_period_instances_length),
    instance_store_size(instance_store_size),
    num_diff_distr_instances(num_diff_distr_instances),
    bbt_pool_size(bbt_pool_size),
    eviction_interval(eviction_interval),
    transfer_kappa_threshold(transfer_kappa_threshold),
    transfer_match_lowerbound(transfer_match_lowerbound),
    gamma(gamma),
    num_trees(num_trees),
    disable_drift_detection(disable_drift_detection) {

    mrand = std::mt19937(seed);

    if (boost_mode_map.find(boost_mode_str) == boost_mode_map.end() ) {
        if (boost_mode_str == "disable_transfer") {
            this->enable_transfer = false;
        } else {
            cout << "Invalid boost mode" << endl;
            exit(1);

        }
    }
    this->boost_mode = boost_mode_map[boost_mode_str];

}

double compute_kappa(deque<int> predicted_labels, deque<int> actual_labels, int class_count) {
    if (predicted_labels.size() != actual_labels.size()) {
        return std::numeric_limits<double>::min();
    }

    // prepare confusion matrix
    vector<vector<int>> confusion_matrix(class_count, vector<int>(class_count, 0));
    int correct = 0;

    for (int i = 0; i < predicted_labels.size(); i++) {
        confusion_matrix[actual_labels[i]][predicted_labels[i]]++;
        if (actual_labels[i] == predicted_labels[i]) {
            correct++;
        }
    }

    double accuracy = (double) correct / predicted_labels.size();

    // computes the Cohen's kappa coefficient
    int sample_count = predicted_labels.size();
    double p0 = accuracy;
    double pc = 0.0;
    int row_count = class_count;
    int col_count = class_count;


    for (int i = 0; i < row_count; i++) {
        double row_sum = 0;
        for (int j = 0; j < col_count; j++) {
            row_sum += confusion_matrix[i][j];
        }

        double col_sum = 0;
        for (int j = 0; j < row_count; j++) {
            col_sum += confusion_matrix[j][i];
        }

        pc += (row_sum / sample_count) * (col_sum / sample_count);
    }

    if (pc == 1) {
        return 1;
    }

    return (p0 - pc) / (1.0 - pc);
}


void trans_tree::init() {
    foreground_tree = make_tree(0);

    bbt_pool = make_unique<boosted_bg_tree_pool>(
            boost_mode,
            bbt_pool_size,
            eviction_interval,
            transfer_kappa_threshold,
            make_tree(-1),
            1);

    foreground_tree->tree_pool_id = tree_pool.size();
    tree_pool.push_back(foreground_tree);
}

shared_ptr<hoeffding_tree> trans_tree::make_tree(int tree_pool_id) {
    return make_shared<hoeffding_tree>(warning_delta, drift_delta, instance_store_size, num_trees, mrand);
}

void trans_tree::train() {
    if (foreground_tree == nullptr) {
        init();
    }

    int actual_label = instance->getLabel();
    if (actual_labels.size() >= kappa_window_size) {
        actual_labels.pop_front();
    }
    actual_labels.push_back(actual_label);

    if (enable_transfer) {
        foreground_tree->store_instance(instance);
        transfer(instance);
    }

    foreground_tree->train(*instance);

    int predicted_label = foreground_tree->predict(*instance, true);
    int error_count = (int) (predicted_label != actual_label);

    bool warning_detected_only = false;
    bool drift_detected = false;

    // detect actual drift
    if (!disable_drift_detection && detect_change(error_count, foreground_tree->drift_detector)) {
        cout << "actual_drift detected-------------------" << endl;
        drift_detected = true;
        foreground_tree->warning_detector->resetChange();
        foreground_tree->drift_detector->resetChange();

        if (foreground_tree->bg_tree == nullptr) {
            foreground_tree = make_tree(-1);
        } else {
            foreground_tree = foreground_tree->bg_tree;
        }
        foreground_tree->tree_pool_id = tree_pool.size();
        tree_pool.push_back(foreground_tree);

        if (bbt_pool == nullptr) {
            shared_ptr<hoeffding_tree> tree_template =
                    make_shared<hoeffding_tree>(*make_tree(-1));
            bbt_pool = make_unique<boosted_bg_tree_pool>(
                    boost_mode,
                    bbt_pool_size,
                    eviction_interval,
                    transfer_kappa_threshold,
                    tree_template,
                    1);
        }
    }

    // detect warning
    if (!disable_drift_detection && detect_change(error_count, foreground_tree->warning_detector)) {
        foreground_tree->bg_tree = make_tree(-1);
        foreground_tree->warning_detector->resetChange();

        if (!drift_detected && enable_transfer) {
            warning_detected_only = true;

            shared_ptr<hoeffding_tree> tree_template =
                    make_shared<hoeffding_tree>(*foreground_tree->bg_tree);
            bbt_pool = make_unique<boosted_bg_tree_pool>(
                    boost_mode,
                    bbt_pool_size,
                    eviction_interval,
                    transfer_kappa_threshold,
                    tree_template,
                    1);
        }
    }
}

bool trans_tree::transfer(Instance* instance) {
    if (bbt_pool == nullptr) {
        return false;
    }

    bool registered_tree_pools_have_concepts = false;
    for (auto registered_tree_pool : registered_tree_pools) {
        if (registered_tree_pool->size() != 0) {
            registered_tree_pools_have_concepts = true;
            break;
        }
    }
    if (!registered_tree_pools_have_concepts) {
        return false;
    }

    bbt_pool->online_boost(instance, true);

    if (bbt_pool->matched_tree == nullptr) {
        // During drift warning period
        bbt_pool->warning_period_instances.push_back(instance);

        if (bbt_pool->warning_period_instances.size() < least_transfer_warning_period_length) {
            // cout << "-------------------------------------warning_period_instances size is not enough: "
            //      << bbt_pool->warning_period_instances.size() << endl;
            // bbt_pool = nullptr;
            return false;
        } else {
            shared_ptr<hoeffding_tree> matched_tree =
                    match_concept(bbt_pool->warning_period_instances);
            if (matched_tree == nullptr) {
                bbt_pool = nullptr;
                return false;
            } else {
                bbt_pool->matched_tree = matched_tree;
            }
        }
    }

    // After tree matching, perform boosting with weight decrement
    for (int j = 0; j < num_diff_distr_instances; j++) {
        Instance* transfer_instance = bbt_pool->get_next_diff_distr_instance();
        if (transfer_instance != nullptr) {
            bbt_pool->online_boost(transfer_instance, false);
        }
    }

    // Attempt transfer
    shared_ptr<hoeffding_tree> transfer_candidate = bbt_pool->get_best_model(actual_labels,
                                                                             instance->getNumberClasses());
    if (transfer_candidate == nullptr) {
        return false;
    }

    foreground_tree->kappa = compute_kappa(foreground_tree->predicted_labels,
                                           actual_labels,
                                           instance->getNumberClasses());

    if (transfer_candidate->kappa - foreground_tree->kappa >= transfer_kappa_threshold
        && transfer_candidate->kappa >= transfer_kappa_threshold) {

        transfer_candidate->tree_pool_id = tree_pool.size();
        tree_pool.push_back(transfer_candidate);

        foreground_tree = transfer_candidate;
        transferred_tree_total_count += 1;
        bbt_pool = nullptr; // TODO reduce overhead
        cout << "transferred tree kappa: " << transfer_candidate->kappa
             << " | "
             << "foreground tree kappa: " << foreground_tree->kappa << endl;
    }

    return true;
}

shared_ptr<hoeffding_tree> trans_tree::match_concept(vector<Instance*> warning_period_instances) {
    shared_ptr<hoeffding_tree> matched_tree = nullptr;
    double highest_kappa = transfer_match_lowerbound;
    int matched_tree_idx = -1;

    // For kappa calculation
    int class_count = warning_period_instances[0]->getNumberClasses();
    deque<int> true_labels;
    for (auto warning_period_instance : warning_period_instances) {
        true_labels.push_back(warning_period_instance->getLabel());
    }

    for (auto registered_tree_pool : registered_tree_pools) {
        for (int i = 0; i < registered_tree_pool->size(); i++) {
            auto trans_tree = (*registered_tree_pool)[i];

            deque<int> predicted_labels;
            for (auto warning_period_instance : warning_period_instances) {
                int prediction = trans_tree->predict(*warning_period_instance, false);
                predicted_labels.push_back(prediction);
            }

            trans_tree->kappa = compute_kappa(predicted_labels, true_labels, class_count);
            cout << "match_concept trans_tree kappa: " << trans_tree->kappa << endl;
            if (highest_kappa < trans_tree->kappa) {
                highest_kappa = trans_tree->kappa;
                matched_tree = trans_tree;
                matched_tree_idx = i;
            }
        }
    }

    if (boost_mode == boost_modes_enum::atradaboost_mode && matched_tree_idx != -1) {
        // bbt_pool->weight_factor = 1.0 / (1.0 + pow(highest_kappa / (1.0 - highest_kappa), -gamma));
        bbt_pool->weight_factor = tanh(gamma * highest_kappa);
    }

    cout << "------------------------------matched_tree_idx: " << matched_tree_idx
         << "| kappa: " << highest_kappa
        << "| weight_factor: " << bbt_pool->weight_factor
         << "| size: " << instance_store_size << endl;

    return matched_tree;
}

void trans_tree::register_tree_pool(vector<shared_ptr<hoeffding_tree>>& _tree_pool) {
    this->registered_tree_pools.push_back(&_tree_pool);
}

vector<shared_ptr<hoeffding_tree>>& trans_tree::get_concept_repo() {
    return this->tree_pool;
}

int trans_tree::predict() {
    if (foreground_tree == nullptr) {
        init();
    }

    return foreground_tree->predict(*instance, true);
}

int trans_tree::get_transferred_tree_group_size() {
    return transferred_tree_total_count;
}

int trans_tree::get_tree_pool_size() {
    return tree_pool.size();
}

bool trans_tree::detect_change(int error_count, unique_ptr<HT::ADWIN>& detector) {
    double old_error = detector->getEstimation();
    bool error_change = detector->setInput(error_count);

    if (!error_change) {
        return false;
    }

    if (old_error > detector->getEstimation()) {
        // error is decreasing
        return false;
    }

    return true;
}

bool trans_tree::init_data_source(const string& filename) {
    cout << "Initializing data source..." << endl;

    reader = make_unique<ArffReader>();
    if (!reader->setFile(filename)) {
        cout << "Failed to open file: " << filename << endl;
        exit(1);
    }

    return true;
}

bool trans_tree::get_next_instance() {
    if (!reader->hasNextInstance()) {
        return false;
    }

    instance = reader->nextInstance();
    return true;
}

int trans_tree::get_cur_instance_label() {
    return instance->getLabel();
}

void trans_tree::delete_cur_instance() {
    delete instance;
}


// class tree
hoeffding_tree::hoeffding_tree(double warning_delta,
                               double drift_delta,
                               int instance_store_size,
                               int num_trees,
                               std::mt19937 mrand) :
        warning_delta(warning_delta),
        drift_delta(drift_delta),
        instance_store_size(instance_store_size),
        num_trees(num_trees),
        mrand(mrand) {

     poisson_distr = std::poisson_distribution<int>(1);

    for (int i = 0; i < num_trees; i++) {
        tree.push_back(make_shared<HT::HoeffdingTree>(mrand));
    }

    warning_detector = make_unique<HT::ADWIN>(warning_delta);
    drift_detector = make_unique<HT::ADWIN>(drift_delta);
    bg_tree = nullptr;
}

hoeffding_tree::hoeffding_tree(hoeffding_tree const &rhs) :
            warning_delta(rhs.warning_delta),
            drift_delta(rhs.drift_delta),
            instance_store_size(rhs.instance_store_size),
            num_trees(rhs.num_trees),
            mrand(rhs.mrand) {

     poisson_distr = std::poisson_distribution<int>(1);

    for (int i = 0; i < num_trees; i++) {
        tree.push_back(make_shared<HT::HoeffdingTree>(mrand));
    }

    warning_detector = make_unique<HT::ADWIN>(warning_delta);
    drift_detector = make_unique<HT::ADWIN>(drift_delta);
    bg_tree = nullptr;
}

int hoeffding_tree::predict(Instance& instance, bool track_prediction) {
     int num_classes = instance.getNumberClasses();
     vector<int> votes(num_classes, 0);
 
     for (int i = 0; i < num_trees; i++) {
         double* classPredictions = tree[i]->getPrediction(instance);
         double max_val = classPredictions[0];
         int predicted_label = 0;
         
         for (int j = 1; j < instance.getNumberClasses(); j++) {
             if (max_val < classPredictions[j]) {
                 max_val = classPredictions[j];
                 predicted_label = j;
             }
         }

         votes[predicted_label]++;
     }

     int result = vote(votes);

    if (track_prediction) {
        if (predicted_labels.size() >= kappa_window_size) {
            predicted_labels.pop_front();
        }
        predicted_labels.push_back(result);
    }
 
     return result;
}

// original
// int hoeffding_tree::predict(Instance& instance, bool track_prediction) {
//     double* classPredictions = tree->getPrediction(instance);
//     int result = 0;
//     double max_val = classPredictions[0];
// 
//     // Find class label with the highest probability
//     for (int i = 1; i < instance.getNumberClasses(); i++) {
//         if (max_val < classPredictions[i]) {
//             max_val = classPredictions[i];
//             result = i;
//         }
//     }
// 
//     if (track_prediction) {
//         if (predicted_labels.size() >= kappa_window_size) {
//             predicted_labels.pop_front();
//         }
//         predicted_labels.push_back(result);
//     }
// 
//     return result;
// }

int hoeffding_tree::vote(const vector<int>& votes) {
    int max_votes = votes[0];
    int predicted_label = 0;

    for (int i = 1; i < votes.size(); i++) {
        if (max_votes < votes[i]) {
            max_votes = votes[i];
            predicted_label = i;
        }
    }

    return predicted_label;
}
 
void hoeffding_tree::train(Instance& instance) {
    for (int i = 0; i < num_trees; i++) {
        int weight = poisson_distr(mrand);
        if (weight == 0) {
            continue;
        }

        instance.setWeight(weight);
        tree[i]->train(instance);
    }

    if (bg_tree != nullptr) {
        bg_tree->train(instance);
    }
}

void hoeffding_tree::store_instance(Instance* instance) {
    if (instance == nullptr) {
        cout << "nullptr instance added! " << endl;
        exit(1);
    }

    if (this->instance_store.size() < this->instance_store_size) {
        this->instance_store.push_back(instance);
        // this->instance_store.pop_front();
    }

    if (this->bg_tree != nullptr) {
        if (bg_tree->instance_store.size() < this->instance_store_size) {
            bg_tree->instance_store.push_back(instance);
            // trans_bg_tree->instance_store.pop_front();
        }
    }
}

// class boosted_bg_tree_pool
trans_tree::boosted_bg_tree_pool::boosted_bg_tree_pool(
        enum boost_modes_enum boost_mode,
        int pool_size,
        int eviction_interval,
        double transfer_kappa_threshold,
        shared_ptr<hoeffding_tree> tree_template,
        int lambda):
        boost_mode(boost_mode),
        eviction_interval(eviction_interval),
        transfer_kappa_threshold(transfer_kappa_threshold),
        pool_size(pool_size),
        tree_template(tree_template),
        lambda(lambda) {

    mrand = std::mt19937(42);

    for (int i = 0; i < pool_size; i++) {
        oob_tree_lam_sum.push_back(0);
        oob_tree_correct_lam_sum.push_back(0);
        oob_tree_wrong_lam_sum.push_back(0);

        lam_sum_correct_src.push_back(0);
        lam_sum_wrong_src.push_back(0);
        error_src.push_back(0);

        lam_sum_correct_tgt.push_back(0);
        lam_sum_wrong_tgt.push_back(0);
        error_tgt.push_back(0);
        weight_distri_tgt.push_back(0);
    }

    // init BBT
    if (boost_mode == boost_modes_enum::no_boost_mode) {
        update_bbt();
    } else {
        for (int i = 0; i < pool_size; i++) {
            update_bbt();
        }
    }
}

void trans_tree::boosted_bg_tree_pool::online_boost(Instance *instance,
                                                     bool is_same_distribution) {
    if (instance == nullptr) {
        cout << "no_boost(): null instance" << endl;
        exit(1);
    }

    if (boost_mode != boost_modes_enum::no_boost_mode) {
        boost_count += 1;
        if (boost_count % eviction_interval == 0) {
            update_bbt();
        }
    }

    switch(boost_mode) {
        case boost_modes_enum::no_boost_mode:
            this->no_boost(instance);
            break;
        case boost_modes_enum::ozaboost_mode:
            this->ozaboost(instance);
            break;
        case boost_modes_enum::tradaboost_mode:
            this->tradaboost(instance, is_same_distribution);
            break;
        case boost_modes_enum::otradaboost_mode:
            this->otradaboost(instance, is_same_distribution);
            break;
        case boost_modes_enum::atradaboost_mode:
            this->atradaboost(instance, is_same_distribution);
            break;
        default:
            cout << "Incorrect boost_mode" << endl;
            exit(1);
    }

    if (is_same_distribution) {
        this->perf_eval(instance);
    }
}

Instance* trans_tree::boosted_bg_tree_pool::get_next_diff_distr_instance() {
    if (instance_store_idx >= matched_tree->instance_store.size()) {
        return nullptr;
    }
    return matched_tree->instance_store[instance_store_idx++];
}

void trans_tree::boosted_bg_tree_pool::perf_eval(Instance* instance) {
    for (int i = 0; i < pool.size(); i++) {
        auto tree = pool[i];
        int predicted_label = tree->predict(*instance, true);
        int error_count = (int) (predicted_label != instance->getLabel());
        if (trans_tree::detect_change(error_count, tree->drift_detector)) {
            pool[i] = std::make_shared<hoeffding_tree>(*tree_template);
        }
        if (trans_tree::detect_change(error_count, tree->warning_detector)) {
            // do nothing
        }
    }
}

shared_ptr<hoeffding_tree> trans_tree::boosted_bg_tree_pool::get_best_model(deque<int> actual_labels,
                                                                         int class_count) {
    shared_ptr<hoeffding_tree> best_model = nullptr;
    double highest_kappa = 0;
    int pool_pos_idx = -1;
    for (int i = 0; i < pool.size(); i++) {
        auto tree = pool[i];
        tree->kappa = compute_kappa(tree->predicted_labels, actual_labels, class_count);
        if (highest_kappa < tree->kappa) {
            highest_kappa = tree->kappa;
            best_model = tree;
            pool_pos_idx = i;
        }
    }

    if (highest_kappa >= transfer_kappa_threshold) {
        cout << "------------------------------pool_pos_idx: " << pool_pos_idx << endl;
    }

    return best_model;
}

void trans_tree::boosted_bg_tree_pool::update_bbt() {
    bbt_counter++;

    // create a new boosting tree for current mini-batch
    shared_ptr<hoeffding_tree> new_tree = std::make_shared<hoeffding_tree>(*tree_template);
    if (pool.size() < pool_size) {
        pool.push_back(new_tree);
    } else {
        pool[bbt_counter % pool_size] = new_tree;
    }
}

void trans_tree::boosted_bg_tree_pool::no_boost(Instance* instance) {
    // Only one tree exists in no_boost_mode
    instance->setWeight(1);
    pool[0]->train(*instance);
}

void trans_tree::boosted_bg_tree_pool::ozaboost(Instance* instance) {
    double lambda_d = 1;
    instance->setWeight(1);

    // vector<double> lambda_vals;

    for (int i = 0; i < pool.size(); i++) {
        auto tree = pool[i];

        // bagging
        std::poisson_distribution<int> poisson_distr(lambda_d);
        double k = poisson_distr(mrand);

        double weight = instance->getWeight();
        // oob_tree_lam_sum[i] += k*weight;
        if (k > 0 && weight > 0) {
            instance->setWeight(k * weight);

            tree->train(*instance);
            instance->setWeight(weight);
        }

        oob_tree_lam_sum[i] += lambda_d;
        bool correctly_classified;
        if (tree->predict(*instance, false) == instance->getLabel()) {
            oob_tree_correct_lam_sum[i] += lambda_d;
            correctly_classified = true;
        } else {
            oob_tree_wrong_lam_sum[i] += lambda_d;
            correctly_classified = false;
        }

        if (correctly_classified) {
            if (oob_tree_correct_lam_sum[i] >= epsilon) {
                lambda_d *= oob_tree_lam_sum[i] / (2 * oob_tree_correct_lam_sum[i]);
            }
        } else {
            if (oob_tree_wrong_lam_sum[i] >= epsilon) {
                lambda_d *= oob_tree_lam_sum[i] / (2 * oob_tree_wrong_lam_sum[i]);
            }
        }
        // lambda_vals.push_back(lambda_d);
    }

    // cout << "lambdas: ";
    // for (auto v : lambda_vals) {
    //     cout << v << " ";
    // }
    // cout << endl;
}

void trans_tree::boosted_bg_tree_pool::tradaboost(Instance* instance, bool is_same_distribution) {
    double lambda_d = 1;
    instance->setWeight(1);
    if (!is_same_distribution) {
        num_src_instances += 1;
    }
    double beta = 1.0 / (1 + sqrt(2 * log(num_src_instances) / pool.size()));

    vector<double> lambda_vals;

    for (int i = 0; i < pool.size(); i++) {
        auto tree = pool[i];

        // bagging
        std::poisson_distribution<int> poisson_distr(lambda_d);
        double k = poisson_distr(mrand);
        if (k > 0) {
            double weight = instance->getWeight();
            instance->setWeight(k * weight);

            tree->train(*instance);
            instance->setWeight(weight);
        }

        if (is_same_distribution) {
            if (tree->predict(*instance, false) == instance->getLabel()) {
                lam_sum_correct_tgt[i] += lambda_d;
            } else {
                lam_sum_wrong_tgt[i] += lambda_d;
            }
        } else {
            if (tree->predict(*instance, false) == instance->getLabel()) {
                lam_sum_correct_src[i] += lambda_d;
            } else {
                lam_sum_wrong_src[i] += lambda_d;
            }
        }

        double lam_sum = lam_sum_correct_tgt[i] + lam_sum_wrong_tgt[i] + lam_sum_correct_src[i] + lam_sum_wrong_src[i];
        error_tgt[i] = lam_sum_wrong_tgt[i] / lam_sum;
        error_src[i] = lam_sum_wrong_src[i] / lam_sum;
        weight_distri_tgt[i] = (lam_sum_correct_tgt[i] + lam_sum_wrong_tgt[i]) / lam_sum;

        double denom = 1 + weight_distri_tgt[i] - (1-beta) * error_src[i] - 2*error_tgt[i];
        if (is_same_distribution) {
            if (tree->predict(*instance, false) == instance->getLabel()) {
                lambda_d = lambda_d / denom;
            } else {
                lambda_d = lambda_d * (weight_distri_tgt[i] - error_tgt[i]) / (error_tgt[i] * denom);
            }
        } else {
            if (tree->predict(*instance, false) == instance->getLabel()) {
                lambda_d = lambda_d / denom;
            } else {
                lambda_d = lambda_d * beta / denom;
            }
        }
    }
}

void trans_tree::boosted_bg_tree_pool::otradaboost(Instance* instance, bool is_same_distribution) {
    double lambda_d = 1;
    instance->setWeight(1);
    if (!is_same_distribution) {
        num_src_instances += 1;
    }
    double beta = 1.0 / (1 + sqrt(2 * log(num_src_instances) / pool.size()));

    vector<double> lambda_vals;

    for (int i = 0; i < pool.size(); i++) {
        auto tree = pool[i];

        // bagging
        std::poisson_distribution<int> poisson_distr(lambda_d);
        double k = poisson_distr(mrand);
        if (k > 0) {
            double weight = instance->getWeight();
            instance->setWeight(k * weight);

            tree->train(*instance);
            instance->setWeight(weight);
        }

        if (k == 0) {
            continue;
        }

        if (is_same_distribution) {
            if (tree->predict(*instance, false) == instance->getLabel()) {
                lam_sum_correct_tgt[i] += lambda_d;
            } else {
                lam_sum_wrong_tgt[i] += lambda_d;
            }
        } else {
            if (tree->predict(*instance, false) == instance->getLabel()) {
                lam_sum_correct_src[i] += lambda_d;
            } else {
                lam_sum_wrong_src[i] += lambda_d;
            }
        }

        double lam_sum = lam_sum_correct_tgt[i] + lam_sum_wrong_tgt[i] + lam_sum_correct_src[i] + lam_sum_wrong_src[i];
        error_tgt[i] = lam_sum_wrong_tgt[i] / lam_sum;
        error_src[i] = lam_sum_wrong_src[i] / lam_sum;
        weight_distri_tgt[i] = (lam_sum_correct_tgt[i] + lam_sum_wrong_tgt[i]) / lam_sum;

        double denom = 1 + weight_distri_tgt[i] - (1-beta) * error_src[i] - 2*error_tgt[i];
        if (is_same_distribution) {
            if (tree->predict(*instance, false) == instance->getLabel()) {
                lambda_d = lambda_d / denom;
            } else {
                lambda_d = lambda_d * (weight_distri_tgt[i] - error_tgt[i]) / (error_tgt[i] * denom);
            }
        } else {
            if (tree->predict(*instance, false) == instance->getLabel()) {
                lambda_d = lambda_d / denom;
            } else {
                lambda_d = lambda_d * beta / denom;
            }
        }
    }
}

void trans_tree::boosted_bg_tree_pool::atradaboost(Instance* instance, bool is_same_distribution) {
    double lambda_d = 1;
    instance->setWeight(1);
    if (!is_same_distribution) {
        num_src_instances += 1;
        // lambda_d *= weight_factor;
    }
    double beta = 1.0 / (1 + sqrt(2 * log(num_src_instances) / pool.size()));

    vector<double> lambda_vals;

    for (int i = 0; i < pool.size(); i++) {
        auto tree = pool[i];

        // bagging
        std::poisson_distribution<int> poisson_distr(lambda_d);
        double k = poisson_distr(mrand);
        if (k > 0) {
            double weight = instance->getWeight();
            instance->setWeight(k * weight);

            tree->train(*instance);
            instance->setWeight(weight);
        }

        if (is_same_distribution) {
            if (tree->predict(*instance, false) == instance->getLabel()) {
                lam_sum_correct_tgt[i] += lambda_d;
            } else {
                lam_sum_wrong_tgt[i] += lambda_d;
            }
        } else {
            if (tree->predict(*instance, false) == instance->getLabel()) {
                lam_sum_correct_src[i] += lambda_d;
            } else {
                lam_sum_wrong_src[i] += lambda_d;
            }
        }

        double lam_sum = lam_sum_correct_tgt[i] + lam_sum_wrong_tgt[i] + lam_sum_correct_src[i] + lam_sum_wrong_src[i];
        error_tgt[i] = lam_sum_wrong_tgt[i] / lam_sum;
        error_src[i] = lam_sum_wrong_src[i] / lam_sum;
        weight_distri_tgt[i] = (lam_sum_correct_tgt[i] + lam_sum_wrong_tgt[i]) / lam_sum;

        double denom = 1 + weight_distri_tgt[i] - (1-beta) * error_src[i] - 2*error_tgt[i];
        if (is_same_distribution) {
            if (tree->predict(*instance, false) == instance->getLabel()) {
                lambda_d = lambda_d / denom;
            } else {
                lambda_d = lambda_d * (weight_distri_tgt[i] - error_tgt[i]) / (error_tgt[i] * denom);
            }
        } else {
            if (tree->predict(*instance, false) == instance->getLabel()) {
                lambda_d = lambda_d / denom;
            } else {
                lambda_d = lambda_d * beta / denom;
            }
            // lambda_d *= (1+weight_factor);
            lambda_d *= weight_factor;
        }
    }
}
