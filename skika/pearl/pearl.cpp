#include "pearl.h"

pearl::pearl(int num_trees,
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
             bool enable_state_graph) :
        num_trees(num_trees),
        max_num_candidate_trees(max_num_candidate_trees),
        repo_size(repo_size),
        edit_distance_threshold(edit_distance_threshold),
        kappa_window_size(kappa_window_size),
        lossy_window_size(lossy_window_size),
        reuse_window_size(reuse_window_size),
        arf_max_features(arf_max_features),
        bg_kappa_threshold(bg_kappa_threshold),
        cd_kappa_threshold(cd_kappa_threshold),
        reuse_rate_upper_bound(reuse_rate_upper_bound),
        warning_delta(warning_delta),
        drift_delta(drift_delta),
        enable_state_adaption(enable_state_adaption),
        enable_state_graph(enable_state_graph) {

    init();
    mrand = std::mt19937(0);
}

void pearl::init() {

    tree_pool = vector<shared_ptr<adaptive_tree>>(num_trees);

    for (int i = 0; i < num_trees; i++) {
        tree_pool[i] = make_adaptive_tree(i);
        adaptive_trees.push_back(tree_pool[i]);
    }

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

shared_ptr<pearl::adaptive_tree> pearl::make_adaptive_tree(int tree_pool_id) {
    return make_shared<adaptive_tree>(tree_pool_id,
                                      kappa_window_size,
                                      warning_delta,
                                      drift_delta);
}

bool pearl::init_data_source(const string& filename) {

    LOG("Initializing data source...");

    reader = make_unique<ArffReader>();

    if (!reader->setFile(filename)) {
        std::cout << "Failed to open file: " << filename << std::endl;
        exit(1);
    }

    return true;
}

void pearl::prepare_instance(Instance& instance) {
    vector<int> attribute_indices;

    // select random features
    for (int i = 0; i < arf_max_features; i++) {
        std::uniform_int_distribution<> uniform_distr(0, num_features);
        int feature_idx = uniform_distr(mrand);
        attribute_indices.push_back(feature_idx);
    }

    instance.setAttributeStatus(attribute_indices);
}

int pearl::predict() {
    int actual_label = instance->getLabel();

    int num_classes = instance->getNumberClasses();
    vector<int> votes(num_classes, 0);

    if (enable_state_adaption) {
        predict_with_state_adaption(votes, actual_label);

    } else {
        predict_basic(votes, actual_label);
    }

    // return vote(votes) == actual_label;
    return vote(votes);
}

int pearl::get_cur_instance_label() {
    return instance->getLabel();
}

void pearl::delete_cur_instance() {
    delete instance;
}

void pearl::predict_with_state_adaption(vector<int>& votes, int actual_label) {

    // keep track of actual labels for candidate tree evaluations
    if (actual_labels.size() >= kappa_window_size) {
        actual_labels.pop_front();
    }
    actual_labels.push_back(actual_label);

    int predicted_label;
    vector<int> warning_tree_pos_list;
    vector<int> drifted_tree_pos_list;

    for (int i = 0; i < num_trees; i++) {

        predicted_label = adaptive_trees[i]->predict(*instance, true);

        votes[predicted_label]++;
        int error_count = (int)(actual_label != predicted_label);

        bool warning_detected_only = false;

        // detect warning
        if (detect_change(error_count, adaptive_trees[i]->warning_detector)) {
            warning_detected_only = false;

            adaptive_trees[i]->bg_adaptive_tree = make_adaptive_tree(-1);
            adaptive_trees[i]->warning_detector->resetChange();
        }

        // detect drift
        if (detect_change(error_count, adaptive_trees[i]->drift_detector)) {
            warning_detected_only = true;
            drifted_tree_pos_list.push_back(i);

            adaptive_trees[i]->drift_detector->resetChange();
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

void pearl::predict_basic(vector<int>& votes, int actual_label) {
    int predicted_label;

    for (int i = 0; i < num_trees; i++) {

        predicted_label = adaptive_trees[i]->predict(*instance, true);

        votes[predicted_label]++;
        int error_count = (int)(actual_label != predicted_label);

        // detect warning
        if (detect_change(error_count, adaptive_trees[i]->warning_detector)) {
            adaptive_trees[i]->bg_adaptive_tree = make_adaptive_tree(-1);
            adaptive_trees[i]->warning_detector->resetChange();
        }

        // detect drift
        if (detect_change(error_count, adaptive_trees[i]->drift_detector)) {
            if (adaptive_trees[i]->bg_adaptive_tree) {
                adaptive_trees[i] = move(adaptive_trees[i]->bg_adaptive_tree);
            } else {
                adaptive_trees[i] = make_adaptive_tree(tree_pool.size());
            }
        }
    }
}

int pearl::vote(const vector<int>& votes) {
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

void pearl::train() {
    for (int i = 0; i < num_trees; i++) {
        online_bagging(*instance, *adaptive_trees[i]);
    }
}

void pearl::online_bagging(Instance& instance, adaptive_tree& tree) {
    prepare_instance(instance);

    std::poisson_distribution<int> poisson_distr(1.0);
    int weight = poisson_distr(mrand);

    while (weight > 0) {
        weight--;
        tree.train(instance);
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
    for (auto warning_tree_pos : warning_tree_pos_list) {
        int warning_tree_id = adaptive_trees[warning_tree_pos]->tree_pool_id;
        int next_id = state_graph->get_next_tree_id(warning_tree_id);

        if (next_id == -1) {
            state_graph->set_is_stable(false);
        } else {
            if (!tree_pool[next_id]->is_candidate) {
                // TODO
                if (candidate_trees.size() >= max_num_candidate_trees) {
                    candidate_trees[0]->is_candidate = false;
                    candidate_trees.pop_front();
                }
                tree_pool[next_id]->is_candidate = true;
                candidate_trees.push_back(tree_pool[next_id]);
            }
        }
    }
}

void pearl::pattern_match_candidate_trees(const vector<int>& warning_tree_pos_list) {
    set<int> ids_to_exclude;

    for (int i = 0; i < warning_tree_pos_list.size(); i++) {
        int tree_pos = warning_tree_pos_list[i];

        if (adaptive_trees[tree_pos]->tree_pool_id == -1) {
            LOG("Error: tree_pool_id is not updated");
            exit(1);
        }

        ids_to_exclude.insert(adaptive_trees[tree_pos]->tree_pool_id);
    }

    set<int> closest_state =
        state_queue->get_closest_state(cur_state, ids_to_exclude);

    if (closest_state.size() == 0) {
        return;
    }

    for (auto i : closest_state) {
        if (cur_state.find(i) == cur_state.end()
            && !tree_pool[i]->is_candidate) {

            if (candidate_trees.size() >= max_num_candidate_trees) {
                candidate_trees[0]->is_candidate = false;
                candidate_trees.pop_front();
            }

            tree_pool[i]->is_candidate = true;
            candidate_trees.push_back(tree_pool[i]);
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
        shared_ptr<adaptive_tree> drifted_tree = adaptive_trees[drifted_pos];
        shared_ptr<adaptive_tree> swap_tree;

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

            shared_ptr<adaptive_tree> bg_tree = drifted_tree->bg_adaptive_tree;

            if (!bg_tree) {
                swap_tree = make_adaptive_tree(tree_pool.size());

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
        adaptive_trees[drifted_pos] = swap_tree;

        drifted_tree->reset();
    }

    state_queue->enqueue(cur_state);

    if (enable_state_graph) {
        graph_switch->update_switch();
    }
}

bool pearl::detect_change(int error_count,
                          unique_ptr<HT::ADWIN>& detector) {

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

bool pearl::get_next_instance() {
    if (!reader->hasNextInstance()) {
        return false;
    }

    instance = reader->nextInstance();

    num_features = instance->getNumberInputAttributes();
    arf_max_features = log2(num_features) + 1;

    return true;
}

bool pearl::compare_kappa(shared_ptr<adaptive_tree>& tree1,
                          shared_ptr<adaptive_tree>& tree2) {
    return tree1->kappa < tree2->kappa;
}

int pearl::get_candidate_tree_group_size() const {
    return candidate_trees.size();
}

int pearl::get_tree_pool_size() const {
    return tree_pool.size();
}

// class adaptive_tree
pearl::adaptive_tree::adaptive_tree(int tree_pool_id,
                                    int kappa_window_size,
                                    double warning_delta,
                                    double drift_delta) :
        tree_pool_id(tree_pool_id),
        kappa_window_size(kappa_window_size),
        warning_delta(warning_delta),
        drift_delta(drift_delta) {

    tree = make_unique<HT::HoeffdingTree>();
    warning_detector = make_unique<HT::ADWIN>(warning_delta);
    drift_detector = make_unique<HT::ADWIN>(drift_delta);
}

int pearl::adaptive_tree::predict(Instance& instance, bool track_performance) {
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
        if (predicted_labels.size() >= kappa_window_size) {
            predicted_labels.pop_front();
        }
        predicted_labels.push_back(result);

        // the background tree performs prediction for performance eval
        if (bg_adaptive_tree) {
            bg_adaptive_tree->predict(instance, track_performance);
        }
    }

    return result;
}

void pearl::adaptive_tree::train(Instance& instance) {
    tree->train(instance);

    if (bg_adaptive_tree) {
        bg_adaptive_tree->train(instance);
    }
}

void pearl::adaptive_tree::update_kappa(const deque<int>& actual_labels, int class_count) {

    if (predicted_labels.size() < kappa_window_size || actual_labels.size() < kappa_window_size) {
        kappa = INT_MIN;
        return;
    }

    vector<vector<int>> confusion_matrix(class_count, vector<int>(class_count, 0));
    int correct = 0;

    for (int i = 0; i < kappa_window_size; i++) {
        confusion_matrix[actual_labels[i]][predicted_labels[i]]++;
        if (actual_labels[i] == predicted_labels[i]) {
            correct++;
        }
    }

    double accuracy = (double) correct / kappa_window_size;

    kappa = compute_kappa(confusion_matrix, accuracy, kappa_window_size, class_count);
}

double pearl::adaptive_tree::compute_kappa(const vector<vector<int>>& confusion_matrix,
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

void pearl::adaptive_tree::reset() {
    bg_adaptive_tree = nullptr;
    is_candidate = false;
    warning_detector->resetChange();
    drift_detector->resetChange();
    predicted_labels.clear();
    kappa = INT_MIN;
}

bool pearl::is_state_graph_stable() {
    return state_graph->get_is_stable();
}
