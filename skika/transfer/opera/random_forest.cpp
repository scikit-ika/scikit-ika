#include "random_forest.h"

random_forest::random_forest(
        int num_trees,
        double lambda,
        std::mt19937& mrand) :
        num_trees(num_trees),
        lambda(lambda),
        mrand(mrand) {}

void random_forest::init() {
    for (int i = 0; i < num_trees; i++) {
        foreground_trees.push_back(make_rf_tree());
    }
}

shared_ptr<rf_tree> random_forest::make_rf_tree() {
    return make_shared<rf_tree>(tree_params, mrand);
}

int random_forest::predict(Instance* instance) {
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

void random_forest::train(Instance* instance) {
    if (foreground_trees.empty()) {
        init();
    }

    std::poisson_distribution<int> poisson_distr(lambda);

    for (int i = 0; i < num_trees; i++) {
        int weight = poisson_distr(mrand);

        if (weight == 0) {
            continue;
        }

        instance->setWeight(weight);
        foreground_trees[i]->train(*instance);
    }
}

int random_forest::vote(const vector<int>& votes) {
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

// class rf_tree
rf_tree::rf_tree(tree_params_t tree_params,
                   std::mt19937& mrand) {

    tree = make_unique<HT::HoeffdingTree>(
                tree_params.grace_period,
                tree_params.split_confidence,
                tree_params.tie_threshold,
                tree_params.binary_splits,
                tree_params.no_pre_prune,
                tree_params.nb_threshold,
                tree_params.leaf_prediction_type,
                mrand);

}

int rf_tree::predict(Instance& instance) {
    double* class_predictions = tree->getPrediction(instance);
    int result = 0;
    double max_val = class_predictions[0];

    // Find class label with the highest probability
    for (int i = 1; i < instance.getNumberClasses(); i++) {
        if (max_val < class_predictions[i]) {
            max_val = class_predictions[i];
            result = i;
        }
    }

    return result;
}

void rf_tree::train(Instance& instance) {
    tree->train(instance);
}
