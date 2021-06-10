#include "nacre.h"

nacre::nacre(int num_trees,
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
                     double stability_delta):
        pearl(num_trees,
              max_num_candidate_trees,
              repo_size,
              edit_distance_threshold,
              kappa_window_size,
              lossy_window_size,
              reuse_window_size,
              arf_max_features,
              lambda,
              seed,
              bg_kappa_threshold,
              cd_kappa_threshold,
              reuse_rate_upper_bound,
              warning_delta,
              drift_delta,
              true,
              true),
        pro_drift_window_size(pro_drift_window_size),
        hybrid_delta(hybrid_delta),
        backtrack_window(backtrack_window),
        stability_delta(stability_delta) {

}

void nacre::init() {
    tree_pool = vector<shared_ptr<pearl_tree>>(num_trees);
    stability_detectors = vector<unique_ptr<HT::ADWIN>>();

    for (int i = 0; i < num_trees; i++) {
        tree_pool[i] = make_pearl_tree(i);
        foreground_trees.push_back(tree_pool[i]);
        stability_detectors.push_back(make_unique<HT::ADWIN>(warning_delta));
    }
}

shared_ptr<pearl_tree> nacre::make_pearl_tree(int tree_pool_id) {
    return make_shared<pearl_tree>(tree_pool_id,
                                   kappa_window_size,
                                   pro_drift_window_size,
                                   warning_delta,
                                   drift_delta,
                                   hybrid_delta,
                                   mrand);
}

// foreground trees make predictions, update votes, keep track of actual labbels
// find warning trees, select candidate trees
// find drifted trees, update potential_drifted_tree_indices
// candidate trees make predictions
void nacre::train() {
    if (foreground_trees.empty()) {
        init();
    }

    potential_drifted_tree_indices.clear();
    stable_tree_indices.clear();

    backtrack_instances.push_back(instance);
    num_instances_seen++;

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
            potential_drifted_tree_indices.insert(i);

            cur_tree->warning_detector->resetChange();
            cur_tree->drift_detector->resetChange();
        }

        if (warning_detected_only) {
            warning_tree_pos_list.push_back(i);
        }

        // detect stability
        int correct_count = (int)(actual_label == predicted_label);
        if (cur_tree->replaced_tree
                && detect_stability(correct_count, stability_detectors[i])) {
            stability_detectors[i] = make_unique<HT::ADWIN>(stability_delta);
            stable_tree_indices.push_back(i);
        }
    }

    for (int i = 0; i < candidate_trees.size(); i++) {
        candidate_trees[i]->predict(*instance, true);
    }

    for (int i = 0; i < predicted_trees.size(); i++) {
        predicted_trees[i]->predict(*instance, true);
    }

    // if warnings are detected, find closest state and update candidate_trees list
    if (warning_tree_pos_list.size() > 0) {
        pearl::select_candidate_trees(warning_tree_pos_list);
    }

    // if actual drifts are detected, swap trees and update cur_state
    if (drifted_tree_pos_list.size() > 0) {
        adapt_state(drifted_tree_pos_list, false);
        // pearl::adapt_state(drifted_tree_pos_list);
    }
}

bool nacre::has_actual_drift(int tree_idx) {
    if (potential_drifted_tree_indices.find(tree_idx) != potential_drifted_tree_indices.end()) {
        return false;

    }

    shared_ptr<pearl_tree> cur_tree
        = static_pointer_cast<pearl_tree>(foreground_trees[tree_idx]);

    return cur_tree->has_actual_drift();
}

vector<int> nacre::adapt_state(
        const vector<int>& drifted_tree_pos_list,
        bool is_proactive) {

    if (is_proactive) {
        return adapt_state_with_proactivity(drifted_tree_pos_list, predicted_trees);
    } else {
        return adapt_state_with_proactivity(drifted_tree_pos_list, candidate_trees);
    }
}

vector<int> nacre::adapt_state_with_proactivity(
        const vector<int>& drifted_tree_pos_list,
        deque<shared_ptr<pearl_tree>>& _candidate_trees) {

    vector<int> actual_drifted_tree_indices; // for return

    int class_count = instance->getNumberClasses();

    // sort candiate trees by kappa
    for (int i = 0; i < _candidate_trees.size(); i++) {
        _candidate_trees[i]->update_kappa(actual_labels, class_count);
    }
    sort(_candidate_trees.begin(), _candidate_trees.end(), compare_kappa);

    for (int i = 0; i < drifted_tree_pos_list.size(); i++) {
        // TODO
        if (tree_pool.size() >= repo_size) {
            std::cout << "tree_pool full: "
                      << std::to_string(tree_pool.size()) << endl;
            exit(1);
        }

        int drifted_pos = drifted_tree_pos_list[i];
        shared_ptr<pearl_tree> drifted_tree = static_pointer_cast<pearl_tree>(foreground_trees[drifted_pos]);
        shared_ptr<pearl_tree> swap_tree = nullptr;

        drifted_tree->update_kappa(actual_labels, class_count);
        // if (drifted_tree->kappa == INT_MIN) {
        //     continue;
        // }

        cur_state.erase(drifted_tree->tree_pool_id);

        bool add_to_repo = false;

        if (_candidate_trees.size() > 0
                && _candidate_trees.back()->kappa
                    - drifted_tree->kappa >= cd_kappa_threshold) {

            actual_drifted_tree_indices.push_back(drifted_tree_pos_list[i]);

            _candidate_trees.back()->is_candidate = false;
            swap_tree = _candidate_trees.back();
            _candidate_trees.pop_back();

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

                actual_drifted_tree_indices.push_back(drifted_tree_pos_list[i]);

            } else {
                // false positive
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

        // keep track of drifted tree
        swap_tree->replaced_tree = drifted_tree;

        // replace drifted_tree with swap tree
        foreground_trees[drifted_pos] = swap_tree;

        drifted_tree->reset();
    }

    state_queue->enqueue(cur_state);

    if (enable_state_graph) {
        graph_switch->update_switch();
    }

    return actual_drifted_tree_indices;
}

int nacre::find_last_actual_drift_point(int tree_idx) {
    if (backtrack_instances.size() > num_max_backtrack_instances) {
        LOG("backtrack_instances has too many data instance");
        exit(1);
    }

    shared_ptr<pearl_tree> swapped_tree;
    swapped_tree = static_pointer_cast<pearl_tree>(foreground_trees[tree_idx]);
    shared_ptr<pearl_tree> drifted_tree = swapped_tree->replaced_tree;
    swapped_tree->replaced_tree = nullptr;

    if (!drifted_tree || !swapped_tree) {
        cout << "Empty drifted or swapped tree" << endl;
        return -1;
    }

    int drift_correct = 0;
    int swap_correct = 0;
    double drifted_tree_accuracy = 0.0;
    double swapped_tree_accuracy = 0.0;

    deque<int> drifted_tree_predictions;
    deque<int> swapped_tree_predictions;

    for (int i = backtrack_instances.size() - 1; i >= 0; i--) {
        if (!backtrack_instances[i]) {
            LOG("cur instance is null!");
            exit(1);
        }

        int drift_predicted_label = drifted_tree->predict(*backtrack_instances[i], false);
        int swap_predicted_label = swapped_tree->predict(*backtrack_instances[i], false);

        int actual_label = instance->getLabel();
        drifted_tree_predictions.push_back((int) (drift_predicted_label == actual_label));
        swapped_tree_predictions.push_back((int) (swap_predicted_label == actual_label));

        drift_correct += drifted_tree_predictions.back();
        swap_correct += swapped_tree_predictions.back();

        if (drifted_tree_predictions.size() >= backtrack_window) {
            drift_correct -= drifted_tree_predictions.front();
            swap_correct -= swapped_tree_predictions.front();
            drifted_tree_predictions.pop_front();
            swapped_tree_predictions.pop_front();

            if (drift_correct >= swap_correct) {
                return backtrack_instances.size() - i;
            }
        }
    }

    return -1;
}

void nacre::set_expected_drift_prob(int tree_idx, double p) {
    shared_ptr<pearl_tree> cur_tree = nullptr;
    cur_tree = static_pointer_cast<pearl_tree>(foreground_trees[tree_idx]);
    cur_tree->set_expected_drift_prob(p);
}

bool nacre::compare_kappa_arf(shared_ptr<arf_tree>& tree1,
                                  shared_ptr<arf_tree>& tree2) {
    shared_ptr<pearl_tree> pearl_tree1 =
        static_pointer_cast<pearl_tree>(tree1);
    shared_ptr<pearl_tree> pearl_tree2 =
        static_pointer_cast<pearl_tree>(tree2);

    return pearl_tree1->kappa < pearl_tree2->kappa;
}

bool nacre::detect_stability(int error_count,
                                 unique_ptr<HT::ADWIN>& detector) {

    double old_error = detector->getEstimation();
    bool error_change = detector->setInput(error_count);

    if (!error_change) {
       return false;
    }

    if (old_error > detector->getEstimation()) {
        // error is decreasing
        // cout << "error is decreasing" << endl;
        return true;
    }

    return false;
}

vector<int> nacre::get_stable_tree_indices() {
    return stable_tree_indices;
}
