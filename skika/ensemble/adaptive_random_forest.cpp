#include "adaptive_random_forest.h"

adaptive_random_forest::adaptive_random_forest(int num_trees,
                                               int arf_max_features,
                                               double warning_delta,
                                               double drift_delta) :
        num_trees(num_trees),
        arf_max_features(arf_max_features),
        warning_delta(warning_delta),
        drift_delta(drift_delta) {

    mrand = std::mt19937(0);
}

void adaptive_random_forest::init() {
    for (int i = 0; i < num_trees; i++) {
        foreground_trees.push_back(make_arf_tree());
    }
}

shared_ptr<arf_tree> adaptive_random_forest::make_arf_tree() {
    return make_shared<arf_tree>(warning_delta,
                                 drift_delta);
}

int adaptive_random_forest::predict() {
    if (foreground_trees.empty()) {
        init();
    }

    int actual_label = instance->getLabel();

    int num_classes = instance->getNumberClasses();
    vector<int> votes(num_classes, 0);

    for (int i = 0; i < num_trees; i++) {

        int predicted_label = foreground_trees[i]->predict(*instance, true);

        votes[predicted_label]++;
        int error_count = (int)(actual_label != predicted_label);

        // detect warning
        if (detect_change(error_count, foreground_trees[i]->warning_detector)) {
            foreground_trees[i]->bg_arf_tree = make_arf_tree();
            foreground_trees[i]->warning_detector->resetChange();
        }

        // detect drift
        if (detect_change(error_count, foreground_trees[i]->drift_detector)) {
            if (foreground_trees[i]->bg_arf_tree) {
                foreground_trees[i] = move(foreground_trees[i]->bg_arf_tree);
            } else {
                foreground_trees[i] = make_arf_tree();
            }
        }
    }

    return vote(votes);
}

void adaptive_random_forest::train() {
    if (foreground_trees.empty()) {
        init();
    }

    for (int i = 0; i < num_trees; i++) {
        online_bagging(*instance, *foreground_trees[i]);
    }
}

void adaptive_random_forest::online_bagging(Instance& instance, arf_tree& tree) {
    prepare_instance(instance);

    std::poisson_distribution<int> poisson_distr(1.0);
    int weight = poisson_distr(mrand);

    while (weight > 0) {
        weight--;
        tree.train(instance);
    }
}

int adaptive_random_forest::vote(const vector<int>& votes) {
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

bool adaptive_random_forest::detect_change(int error_count,
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

bool adaptive_random_forest::init_data_source(const string& filename) {

    LOG("Initializing data source...");

    reader = make_unique<ArffReader>();

    if (!reader->setFile(filename)) {
        std::cout << "Failed to open file: " << filename << std::endl;
        exit(1);
    }

    return true;
}

void adaptive_random_forest::prepare_instance(Instance& instance) {
    vector<int> attribute_indices;

    // select random features
    for (int i = 0; i < arf_max_features; i++) {
        std::uniform_int_distribution<> uniform_distr(0, num_features);
        int feature_idx = uniform_distr(mrand);
        attribute_indices.push_back(feature_idx);
    }

    instance.setAttributeStatus(attribute_indices);
}

bool adaptive_random_forest::get_next_instance() {
    if (!reader->hasNextInstance()) {
        return false;
    }

    instance = reader->nextInstance();

    num_features = instance->getNumberInputAttributes();
    arf_max_features = log2(num_features) + 1;

    return true;
}

int adaptive_random_forest::get_cur_instance_label() {
    return instance->getLabel();
}

void adaptive_random_forest::delete_cur_instance() {
    delete instance;
}


// class arf_tree
arf_tree::arf_tree(double warning_delta,
                   double drift_delta) :
        warning_delta(warning_delta),
        drift_delta(drift_delta) {

    tree = make_unique<HT::HoeffdingTree>();
    warning_detector = make_unique<HT::ADWIN>(warning_delta);
    drift_detector = make_unique<HT::ADWIN>(drift_delta);
}

int arf_tree::predict(Instance& instance, bool track_performance) {
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

    return result;
}

void arf_tree::train(Instance& instance) {
    tree->train(instance);

    if (bg_arf_tree) {
        bg_arf_tree->train(instance);
    }
}

void arf_tree::reset() {
    bg_arf_tree = nullptr;
    warning_detector->resetChange();
    drift_detector->resetChange();
}
