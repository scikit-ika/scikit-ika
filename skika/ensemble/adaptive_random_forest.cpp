#include "adaptive_random_forest.h"

adaptive_random_forest::adaptive_random_forest(int num_trees,
                                               int arf_max_features,
                                               int lambda,
                                               int seed,
                                               double warning_delta,
                                               double drift_delta) :
        num_trees(num_trees),
        arf_max_features(arf_max_features),
        lambda(lambda),
        warning_delta(warning_delta),
        drift_delta(drift_delta) {

    mrand = std::mt19937(seed);
}

void adaptive_random_forest::init() {
    for (int i = 0; i < num_trees; i++) {
        foreground_trees.push_back(make_arf_tree());
    }
}

shared_ptr<arf_tree> adaptive_random_forest::make_arf_tree() {
    return make_shared<arf_tree>(warning_delta,
                                 drift_delta,
                                 mrand);
}

int adaptive_random_forest::predict() {
    if (foreground_trees.empty()) {
        init();
    }

    int num_classes = instance->getNumberClasses();
    vector<int> votes(num_classes, 0);

    for (int i = 0; i < num_trees; i++) {
        int predicted_label = foreground_trees[i]->predict(*instance);
        votes[predicted_label]++;
    }

    return vote(votes);
}

void adaptive_random_forest::train() {
    if (foreground_trees.empty()) {
        init();
    }

    for (int i = 0; i < num_trees; i++) {
        std::poisson_distribution<int> poisson_distr(lambda);
        int weight = poisson_distr(mrand);

        if (weight == 0) {
            continue;
        }

        instance->setWeight(weight);

        foreground_trees[i]->train(*instance);

        int predicted_label = foreground_trees[i]->predict(*instance);
        int error_count = (int)(predicted_label != instance->getLabel());

        // detect warning
        if (detect_change(error_count, foreground_trees[i]->warning_detector)) {
            foreground_trees[i]->bg_arf_tree = make_arf_tree();
            foreground_trees[i]->warning_detector->resetChange();
        }

        // detect drift
        if (detect_change(error_count, foreground_trees[i]->drift_detector)) {
            if (foreground_trees[i]->bg_arf_tree) {
                foreground_trees[i] = foreground_trees[i]->bg_arf_tree;
            } else {
                foreground_trees[i] = make_arf_tree();
                foreground_trees[i]->bg_arf_tree = nullptr;
            }
            foreground_trees[i]->bg_arf_tree = nullptr;
        }
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

bool adaptive_random_forest::get_next_instance() {
    if (!reader->hasNextInstance()) {
        return false;
    }

    instance = reader->nextInstance();

    num_features = instance->getNumberInputAttributes();
    arf_max_features = sqrt(num_features) + 1;

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
                   double drift_delta,
                   std::mt19937 mrand) :
        warning_delta(warning_delta),
        drift_delta(drift_delta),
        mrand(mrand) {

    tree = make_unique<HT::HoeffdingTree>(mrand);
    warning_detector = make_unique<HT::ADWIN>(warning_delta);
    drift_detector = make_unique<HT::ADWIN>(drift_delta);
    bg_arf_tree = nullptr;
}

int arf_tree::predict(Instance& instance) {
    double* classPredictions = tree->getPrediction(instance);
    int result = 0;
    double max_val = classPredictions[0];

    // Find class label with the highest probability
    for (int i = 1; i < instance.getNumberClasses(); i++) {
        if (max_val < classPredictions[i]) {
            max_val = classPredictions[i];
            result = i;
        }
    }

    return result;
}

void arf_tree::train(Instance& instance) {
    tree->train(instance);

    if (bg_arf_tree != nullptr) {
        bg_arf_tree->train(instance);
    }
}
