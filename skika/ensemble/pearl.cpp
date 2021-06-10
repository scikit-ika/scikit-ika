#include "pearl.h"

pearl::pearl(int num_trees,
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
             bool enable_state_graph) :
        adaptive_random_forest(num_trees,
                               arf_max_features,
                               lambda,
                               seed,
                               warning_delta,
                               drift_delta),
        max_num_candidate_trees(max_num_candidate_trees),
        repo_size(repo_size),
        edit_distance_threshold(edit_distance_threshold),
        kappa_window_size(kappa_window_size),
        lossy_window_size(lossy_window_size),
        reuse_window_size(reuse_window_size),
        bg_kappa_threshold(bg_kappa_threshold),
        cd_kappa_threshold(cd_kappa_threshold),
        reuse_rate_upper_bound(reuse_rate_upper_bound),
        enable_state_adaption(enable_state_adaption),
        enable_state_graph(enable_state_graph) {

    // initialize LRU state pattern queue
    state_queue = make_unique<lru_state>(10000000, edit_distance_threshold); // TODO

    cur_state = set<int>();
    for (int i = 0; i < num_trees; i++) {
        cur_state.insert(i);
    }

    state_queue->enqueue(cur_state);

    // initialize state graph with lossy counting
    state_graph = make_shared<lossy_state_graph>(repo_size,
                                                 lossy_window_size,
                                                 mrand);

    // graph_switch keeps track of tree reuse rate and turns on/off state_graph
    graph_switch = make_unique<state_graph_switch>(state_graph,
                                                   reuse_window_size,
                                                   reuse_rate_upper_bound);

}

void pearl::init() {
    tree_pool = vector<shared_ptr<pearl_tree>>(num_trees);

    for (int i = 0; i < num_trees; i++) {
        tree_pool[i] = make_pearl_tree(i);
        foreground_trees.push_back(tree_pool[i]);
    }
}

shared_ptr<pearl_tree> pearl::make_pearl_tree(int tree_pool_id) {
    return make_shared<pearl_tree>(tree_pool_id,
                                   kappa_window_size,
                                   warning_delta,
                                   drift_delta,
                                   mrand);
}

void pearl::train() {
    if (foreground_trees.empty()) {
        init();
    }

    int actual_label = instance->getLabel();

    // keep track of actual labels for candidate tree evaluations
    if (actual_labels.size() >= kappa_window_size) {
        actual_labels.pop_front();
    }
    actual_labels.push_back(actual_label);

    vector<int> warning_tree_pos_list;
    vector<int> drifted_tree_pos_list;

    shared_ptr<pearl_tree> cur_tree = nullptr;

    for (int i = 0; i < num_trees; i++) {
        std::poisson_distribution<int> poisson_distr(lambda);
        int weight = poisson_distr(mrand);

        if (weight == 0) {
            continue;
        }

        instance->setWeight(weight);

        cur_tree = static_pointer_cast<pearl_tree>(foreground_trees[i]);
        cur_tree->train(*instance);

        int predicted_label = cur_tree->predict(*instance, true);
        int error_count = (int)(predicted_label != actual_label);

        bool warning_detected_only = false;

        // detect warning
        if (detect_change(error_count, cur_tree->warning_detector)) {
            warning_detected_only = true;
            cur_tree->bg_pearl_tree = make_pearl_tree(-1);
            cur_tree->warning_detector->resetChange();
        }

        // detect drift
        if (detect_change(error_count, cur_tree->drift_detector)) {
            warning_detected_only = false;
            drifted_tree_pos_list.push_back(i);

            cur_tree->warning_detector->resetChange();
            cur_tree->drift_detector->resetChange();
        }

        if (warning_detected_only) {
            warning_tree_pos_list.push_back(i);
        }
    }

    for (int i = 0; i < candidate_trees.size(); i++) {
        candidate_trees[i]->predict(*instance, true);
    }

    // if warnings are detected, find closest state and update candidate_trees list
    if (warning_tree_pos_list.size() > 0) {
        select_candidate_trees(warning_tree_pos_list);
    }

    // if actual drifts are detected, swap trees and update cur_state
    if (drifted_tree_pos_list.size() > 0) {
        adapt_state(drifted_tree_pos_list);
    }
}

void pearl::select_candidate_trees(const vector<int>& warning_tree_pos_list) {

    if (enable_state_graph) {
        // try trigger lossy counting
        if (state_graph->update(warning_tree_pos_list.size())) {
            // TODO log
        }
    }

    // add selected neighbors as candidate trees if graph is stable
    if (state_graph->get_is_stable()) {
        tree_transition(warning_tree_pos_list);
    }

    // trigger pattern matching if graph has become unstable
    if (!state_graph->get_is_stable()) {
        pattern_match_candidate_trees(warning_tree_pos_list);

    } else {
        // TODO log
    }
}

void pearl::tree_transition(const vector<int>& warning_tree_pos_list) {
    tree_transition(warning_tree_pos_list, candidate_trees);
}

void pearl::tree_transition(
        const vector<int>& warning_tree_pos_list,
        deque<shared_ptr<pearl_tree>>& _candidate_trees) {
    shared_ptr<pearl_tree> cur_tree = nullptr;
    for (auto warning_tree_pos : warning_tree_pos_list) {
        cur_tree = static_pointer_cast<pearl_tree>(foreground_trees[warning_tree_pos]);

        int warning_tree_id = cur_tree->tree_pool_id;
        int next_id = state_graph->get_next_tree_id(warning_tree_id);

        if (next_id == -1) {
            state_graph->set_is_stable(false);
        } else {
            if (!tree_pool[next_id]->is_candidate) {
                // TODO
                if (_candidate_trees.size() >= max_num_candidate_trees) {
                    _candidate_trees[0]->is_candidate = false;
                    _candidate_trees.pop_front();
                }
                tree_pool[next_id]->is_candidate = true;
                _candidate_trees.push_back(tree_pool[next_id]);
            }
        }
    }
}

void pearl::pattern_match_candidate_trees(const vector<int>& warning_tree_pos_list) {
    pattern_match_candidate_trees(warning_tree_pos_list, candidate_trees);
}

void pearl::pattern_match_candidate_trees(
        const vector<int>& warning_tree_pos_list,
        deque<shared_ptr<pearl_tree>>& _candidate_trees) {

    set<int> ids_to_exclude;
    shared_ptr<pearl_tree> cur_tree = nullptr;

    for (int tree_pos : warning_tree_pos_list) {
        cur_tree = static_pointer_cast<pearl_tree>(foreground_trees[tree_pos]);

        if (cur_tree->tree_pool_id == -1) {
            LOG("Error: tree_pool_id is not updated");
            exit(1);
        }

        ids_to_exclude.insert(cur_tree->tree_pool_id);
    }

    set<int> closest_state =
        state_queue->get_closest_state(cur_state, ids_to_exclude);

    if (closest_state.size() == 0) {
        return;
    }

    for (auto i : closest_state) {
        if (cur_state.find(i) == cur_state.end()
            && !tree_pool[i]->is_candidate) {

            if (_candidate_trees.size() >= max_num_candidate_trees) {
                _candidate_trees[0]->is_candidate = false;
                _candidate_trees.pop_front();
            }

            tree_pool[i]->is_candidate = true;
            _candidate_trees.push_back(tree_pool[i]);
        }
    }
}

void pearl::adapt_state(const vector<int>& drifted_tree_pos_list) {
    int class_count = instance->getNumberClasses();

    // sort candiate trees by kappa
    for (int i = 0; i < candidate_trees.size(); i++) {
        candidate_trees[i]->update_kappa(actual_labels, class_count);
    }
    sort(candidate_trees.begin(), candidate_trees.end(), compare_kappa);

    for (int i = 0; i < drifted_tree_pos_list.size(); i++) {
        // TODO
        if (tree_pool.size() >= repo_size) {
            std::cout << "tree_pool full: "
                      << std::to_string(tree_pool.size()) << endl;
            exit(1);
        }

        int drifted_pos = drifted_tree_pos_list[i];
        shared_ptr<pearl_tree> drifted_tree =
            static_pointer_cast<pearl_tree>(foreground_trees[drifted_pos]);
        shared_ptr<pearl_tree> swap_tree = nullptr;

        drifted_tree->update_kappa(actual_labels, class_count);

        cur_state.erase(drifted_tree->tree_pool_id);

        bool add_to_repo = false;

        if (candidate_trees.size() > 0
            && candidate_trees.back()->kappa
                - drifted_tree->kappa >= cd_kappa_threshold) {
            candidate_trees.back()->is_candidate = false;
            swap_tree = candidate_trees.back();
            candidate_trees.pop_back();

            if (enable_state_graph) {
                graph_switch->update_reuse_count(1);
            }
        }

        if (swap_tree == nullptr) {
            add_to_repo = true;

            if (enable_state_graph) {
                graph_switch->update_reuse_count(0);
            }

            shared_ptr<pearl_tree> bg_tree = drifted_tree->bg_pearl_tree;

            if (!bg_tree) {
                swap_tree = make_pearl_tree(tree_pool.size());

            } else {
                bg_tree->update_kappa(actual_labels, class_count);

                if (bg_tree->kappa == INT_MIN) {
                    // add bg tree to the repo even if it didn't fill the window

                } else if (bg_tree->kappa - drifted_tree->kappa >= bg_kappa_threshold) {

                } else {
                    // false positive
                    add_to_repo = false;

                }

                swap_tree = bg_tree;
            }

            if (add_to_repo) {
                swap_tree->reset();

                // assign a new tree_pool_id for background tree
                // and allocate a slot for background tree in tree_pool
                swap_tree->tree_pool_id = tree_pool.size();
                tree_pool.push_back(swap_tree);

            } else {
                swap_tree->tree_pool_id = drifted_tree->tree_pool_id;

                // TODO
                // swap_tree = move(drifted_tree);
            }
        }

        if (!swap_tree) {
            LOG("swap_tree is nullptr");
            exit(1);
        }

        if (enable_state_graph) {
            state_graph->add_edge(drifted_tree->tree_pool_id, swap_tree->tree_pool_id);
        }

        cur_state.insert(swap_tree->tree_pool_id);

        // replace drifted_tree with swap tree
        foreground_trees[drifted_pos] = swap_tree;

        drifted_tree->reset();
    }

    state_queue->enqueue(cur_state);

    if (enable_state_graph) {
        graph_switch->update_switch();
    }
}

bool pearl::compare_kappa(shared_ptr<pearl_tree>& tree1,
                          shared_ptr<pearl_tree>& tree2) {
    return tree1->kappa < tree2->kappa;
}

int pearl::get_candidate_tree_group_size() const {
    return candidate_trees.size();
}

int pearl::get_tree_pool_size() const {
    return tree_pool.size();
}

bool pearl::is_state_graph_stable() {
    return state_graph->get_is_stable();
}

// class pearl_tree
pearl_tree::pearl_tree(int tree_pool_id,
                       int kappa_window_size,
                       double warning_delta,
                       double drift_delta,
                       mt19937 mrand) :
        arf_tree(warning_delta, drift_delta, mrand),
        tree_pool_id(tree_pool_id),
        kappa_window_size(kappa_window_size) {

    tree = make_unique<HT::HoeffdingTree>(mrand);
    warning_detector = make_unique<HT::ADWIN>(warning_delta);
    drift_detector = make_unique<HT::ADWIN>(drift_delta);
}

pearl_tree::pearl_tree(int tree_pool_id,
                       int kappa_window_size,
                       int pro_drift_window_size,
                       double warning_delta,
                       double drift_delta,
                       double hybrid_delta,
                       std::mt19937 mrand) :
        arf_tree(warning_delta, drift_delta, mrand),
        tree_pool_id(tree_pool_id),
        kappa_window_size(kappa_window_size),
        pro_drift_window_size(pro_drift_window_size),
        hybrid_delta(hybrid_delta) {

    tree = make_unique<HT::HoeffdingTree>(mrand);
    warning_detector = make_unique<HT::ADWIN>(warning_delta);
    drift_detector = make_unique<HT::ADWIN>(drift_delta);
}

int pearl_tree::predict(Instance& instance, bool track_performance) {
    double numberClasses = instance.getNumberClasses();
    double* classPredictions = tree->getPrediction(instance);
    int result = 0;
    double max = classPredictions[0];

    // Find class label with the highest probability
    for (int i = 1; i < numberClasses; i++) {
        if (max < classPredictions[i]) {
            max = classPredictions[i];
            result = i;
        }
    }

    if (track_performance) {

        if (predicted_labels_window.size() >= kappa_window_size) {
            predicted_labels_window.pop_front();
        }
        predicted_labels_window.push_back(result);


        if (pro_drift_window_size > 0) {
            int correct_count = (int) (result == instance.getLabel());
            if (predicted_result_right_window.size() >= pro_drift_window_size) {
                int right_front_label = predicted_result_right_window.front();
                predicted_result_right_window.pop_front();
                right_correct_count -= right_front_label;

                if (predicted_result_left_window.size() >= pro_drift_window_size) {
                    left_correct_count -= predicted_result_left_window.front();
                    predicted_result_left_window.pop_front();
                }
                left_correct_count += right_front_label;
                predicted_result_left_window.push_back(right_front_label);
            }
            right_correct_count += correct_count;
            predicted_result_right_window.push_back(correct_count);
        }

        // the background tree performs prediction for performance eval
        if (bg_pearl_tree) {
            bg_pearl_tree->predict(instance, track_performance);
        }
    }

    return result;
}

void pearl_tree::train(Instance& instance) {
    tree->train(instance);

    if (bg_pearl_tree) {
        bg_pearl_tree->train(instance);
    }
}

double pearl_tree::get_variance() {
    double avg =
        (left_correct_count + right_correct_count) / (pro_drift_window_size * 2);
    double sum = 0.0;

    for (int v : predicted_result_right_window) {
        sum += ((v - avg) * (v - avg));
    }

    for (int v : predicted_result_left_window) {
        sum += ((v - avg) * (v - avg));
    }

    return sum / (pro_drift_window_size * 2);
}

bool pearl_tree::has_actual_drift() {
    if (predicted_result_left_window.size() < pro_drift_window_size) {
        return false;
    }

    double left_window_mean = left_correct_count / pro_drift_window_size;
    double right_window_mean = right_correct_count / pro_drift_window_size;

    double bound = compute_hoeffding_bound(get_variance(), pro_drift_window_size, drift_delta);
    double hybrid_bound = (bound - bound*hybrid_delta) / (hybrid_delta - 2 * abs(bound) * hybrid_delta + 1);

    cout << to_string(right_window_mean) << " - "
         << to_string(left_window_mean) << ">"
         << to_string(bound)
         << endl;

    return (left_window_mean - right_window_mean) > hybrid_bound;
}

double pearl_tree::compute_hoeffding_bound(double variance, double window_size, double delta) {
    // hoeffding bound with bonferroni correction
    double m = 1.0 / ((1.0 / window_size) + (1.0 / window_size));
    delta = delta / (window_size * 2);

    double epsilon = sqrt((2 / m * variance * log(2 / delta)))
                        + 2 / (3 * m) * log(2 / delta);

    return epsilon;
}

void pearl_tree::update_kappa(const deque<int>& actual_labels, int class_count) {

    if (predicted_labels_window.size() < kappa_window_size || actual_labels.size() < kappa_window_size) {
        kappa = INT_MIN;
        return;
    }

    vector<vector<int>> confusion_matrix(class_count, vector<int>(class_count, 0));
    int correct = 0;

    for (int i = 0; i < kappa_window_size; i++) {
        confusion_matrix[actual_labels[i]][predicted_labels_window[i]]++;
        if (actual_labels[i] == predicted_labels_window[i]) {
            correct++;
        }
    }

    double accuracy = (double) correct / kappa_window_size;

    kappa = compute_kappa(confusion_matrix, accuracy, kappa_window_size, class_count);
}

double pearl_tree::compute_kappa(const vector<vector<int>>& confusion_matrix,
                                 double accuracy,
                                 int sample_count,
                                 int class_count) {
    // computes the Cohen's kappa coefficient
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

void pearl_tree::reset() {
    bg_pearl_tree = nullptr;
    replaced_tree = nullptr;
    is_candidate = false;
    warning_detector->resetChange();
    drift_detector->resetChange();
    predicted_result_right_window.clear();
    predicted_result_left_window.clear();
    predicted_labels_window.clear();
    left_correct_count = 0.0;
    right_correct_count = 0.0;
    kappa = INT_MIN;
}

void pearl_tree::set_expected_drift_prob(double p) {
    warning_detector->setExpectedDriftProb(p);
    drift_detector->setExpectedDriftProb(p);
}
