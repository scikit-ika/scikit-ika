#include "opera.h"

opera::opera(
        int seed,
        int num_trees,
        double lambda,
        // transfer learning params
        int num_phantom_branches,
        int squashing_delta,
        int obs_period,
        double conv_delta,
        double conv_threshold,
        int obs_window_size,
        int perf_window_size,
        int min_obs_period,
        int split_range,
        bool grow_transfer_surrogate_during_obs,
        bool force_disable_patching,
        bool force_enable_patching) :
        num_trees(num_trees),
        lambda(lambda),
        num_phantom_branches(num_phantom_branches),
        squashing_delta(squashing_delta),
        obs_period(obs_period),
        conv_delta(conv_delta),
        conv_threshold(conv_threshold),
        obs_window_size(obs_window_size),
        perf_window_size(perf_window_size),
        min_obs_period(min_obs_period),
        split_range(split_range),
        grow_transfer_surrogate_during_obs(grow_transfer_surrogate_during_obs),
        force_disable_patching(force_disable_patching),
        force_enable_patching(force_enable_patching) {

    mrand = std::mt19937(seed);
}


void opera::init() {
    classifier = make_tree();
    tree_pool.push_back(classifier);
}

shared_ptr<random_forest> opera::make_tree() {
    return make_shared<random_forest>(num_trees, lambda, mrand);
}

unique_ptr<phantom_tree> opera::make_phantom_tree() {
    return make_unique<phantom_tree>(
            num_phantom_branches,
            squashing_delta,
            split_range,
            mrand);
}

void opera::train() {
    if (this->classifier == nullptr) {
        // try transfer at the beginning of the stream
        if (transfer_model()) {
            cout << "model transferred" << endl;

        } else {
            init();
        }
    }

    int actual_label = (int) instance->getLabel();
    if (this->actual_labels.size() >= this->perf_window_size) {
        this->actual_labels.pop_front();
    }
    actual_labels.push_back(actual_label);
    int predicted_label = classifier->predict(instance);
    int error_count = (int) (predicted_label != actual_label);

    if (this->grow_transfer_surrogate_during_obs
            && this->transferSurrogate != nullptr) {
        this->transferSurrogate->train(instance);
    }

    if (this->patchClassifier != nullptr) {
        // train transferred model
        this->classifier->train(instance);
        // update transferred model performance
        if (this->transErrorWindow.size() > this->perf_window_size) {
            this->transErrorWindowSum -= this->transErrorWindow.front();
            this->transErrorWindow.pop_front();
        }
        this->transErrorWindow.push_back(error_count);
        this->transErrorWindowSum += error_count;

        // train patch
        int patchErrorCount = error_count;
        if (error_count == 1) {
            // keep track of patch to either turn on/off patch prediction
            if (this->patchClassifier->predict(instance) != instance->getLabel()) {
                patchErrorCount = 1;
            }

            this->patchClassifier->train(instance);
        }

        // update patch performance
        if (this->patchErrorWindow.size() > this->perf_window_size) {
            this->patchErrorWindowSum -= this->patchErrorWindow.front();
            this->patchErrorWindow.pop_front();
        }
        this->patchErrorWindow.push_back(patchErrorCount);
        this->patchErrorWindowSum += patchErrorCount;

        // train new classifier
        int newErrorCount = this->newClassifier->predict(instance) == instance->getLabel() ? 0 : 1;
        this->newClassifier->train(instance);
        // update new classifier performance
        if (this->newErrorWindow.size() > this->perf_window_size){
            this->newErrorWindowSum -= this->newErrorWindow.front();
            this->newErrorWindow.pop_front();
        }
        this->newErrorWindow.push_back(newErrorCount);
        this->newErrorWindowSum += newErrorCount;

    } else if (!this->inObsPeriod) {
        // force_disable_patching. patchClassifier is nullptr.
        this->classifier->train(instance);

    } else {
        // in observation period
        this->obsInstanceStore.push_back(instance);
        this->obsPredictionResults.push_back(error_count);
        if (error_count == 1) {
            this->errorRegionInstanceStore.push_back(instance);
        } else {
            this->aproposRegionInstanceStore.push_back(instance);
        }

        if (!this->m_true_error->is_stable(error_count)) {
            return;
        }

        if (this->min_obs_period > -1 && this->obsInstanceStore.size() < this->min_obs_period) {
            return;
        }

        // out of observation period
        this->inObsPeriod = false;

        if (this->grow_transfer_surrogate_during_obs) {
            this->transferSurrogate = nullptr;
        }

        // start complexity evaluation
        cout << "instance store size: " << obsInstanceStore.size() << endl;

        bool enableAdaptation = false;
        if (this->force_disable_patching) {

        } else if (this->force_enable_patching) {
            enableAdaptation = true;
        } else {
            // theorem
            enableAdaptation = is_high_adaptability();
        }

        if (enableAdaptation) {
            this->errorRegionClassifier = make_tree();
            this->patchClassifier = make_tree();
            this->newClassifier = make_tree();

            for (int idx = 0; idx < this->obsInstanceStore.size(); idx++) {
                Instance* obsInstance = this->obsInstanceStore[idx];
                this->newClassifier->train(obsInstance);

                // TODO copy instance, check if attribute insertion is correct
                Instance* newInstance = cloneInstance(obsInstance);
                vector<double> attribute_values = newInstance->getValues();
                attribute_values.push_back(obsInstance->getLabel());
                // for (auto attribute_value : attribute_values) {
                //     cout << attribute_value << " ";
                // }
                // cout << endl;
                newInstance->addValues(attribute_values);

                this->classifier->train(obsInstance);
                if (this->obsPredictionResults[idx] == 1) {
                    this->patchClassifier->train(obsInstance);
                    newInstance->setLabel(0, 1);
                } else  {
                    newInstance->setLabel(0, 0);
                }

                this->errorRegionClassifier->train(newInstance);
            }
        } else {
            this->classifier = make_tree();
            for (auto obsInstance : this->obsInstanceStore) {
                this->classifier->train(obsInstance);
            }
        }

        this->obsInstanceStore.clear();
        this->errorRegionInstanceStore.clear();
        this->aproposRegionInstanceStore.clear();

    }
}

bool opera::transfer_model() {
    bool registered_tree_pools_have_concepts = false;
    for (auto registered_tree_pool : registered_tree_pools) {
        if (!registered_tree_pool->empty()) {
            registered_tree_pools_have_concepts = true;
            // TODO
            // simply select the first model to transfer
            this->classifier = (*registered_tree_pool)[0];
            break;
        }
    }
    if (!registered_tree_pools_have_concepts) {
        return false;
    }

    if (this->grow_transfer_surrogate_during_obs) {
        this->transferSurrogate = make_shared<random_forest>(*this->classifier); // copy
    }

    this->inObsPeriod = true;
    this->m_true_error = make_unique<true_error>(
            this->obs_window_size,
            this->conv_delta,
            this->conv_threshold,
            this->mrand);

    return true;
}

bool opera::is_high_adaptability() {
    unique_ptr<phantom_tree> full_tree = make_phantom_tree();
    unique_ptr<phantom_tree> error_tree = make_phantom_tree();
    unique_ptr<phantom_tree> correct_tree = make_phantom_tree();

    double f = full_tree->get_construction_complexity(this->obsInstanceStore);
    double e = error_tree->get_construction_complexity(this->errorRegionInstanceStore);
    double c = correct_tree->get_construction_complexity(this->aproposRegionInstanceStore);

    this->full_region_complexity = f;
    this->error_region_complexity = e;
    this->correct_region_complexity = c;

    if ((f < (c || e))
            || (c < e && f < e)
            || (c < e && (e < f && f < c + e))) {
        cout << "low adaptability" << endl;
        return false;
    }

    cout << "high adaptability" << endl;
    return true;
}


void opera::register_tree_pool(vector<shared_ptr<random_forest>>& _tree_pool) {
    this->registered_tree_pools.push_back(&_tree_pool);
}

vector<shared_ptr<random_forest>>& opera::get_concept_repo() {
    return this->tree_pool;
}

int opera::predict() {
    if (this->classifier == nullptr) {
        // try transfer at the beginning of the stream
        if (transfer_model()) {
            cout << "model transferred" << endl;
        } else {
            init();
        }
    }

    if (this->grow_transfer_surrogate_during_obs
            && this->transferSurrogate != nullptr) {
        return this->transferSurrogate->predict(instance);
    }

    if (this->patchClassifier == nullptr) {
        return this->classifier->predict(instance);
    }

    //// new classifier
    if (switch_to_new_classifier()) {
        return this->newClassifier->predict(instance);
    }

    // patch on/off with transferred model
    auto newInstance = (DenseInstance*) cloneInstance(instance);
    vector<double> attribute_values = newInstance->getValues();
    attribute_values.push_back(instance->getLabel());
    newInstance->addValues(attribute_values);

    if ((int) this->errorRegionClassifier->predict(newInstance) == 1) {
        // in error region, check patch performance
        if (turn_on_patch_prediction()) {
            return this->patchClassifier->predict(instance);
        }
    }

    return this->classifier->predict(instance);
}

bool opera::switch_to_new_classifier() {
    if (this->newErrorWindow.size() < this->perf_window_size) {
        return false;
    }

    if (this->newErrorWindowSum < this->patchErrorWindowSum
            && this->newErrorWindowSum < this->transErrorWindowSum) {
        // cout << "switching to the new classifier" << endl;
        return true;
    }

    return false;
}

bool opera::turn_on_patch_prediction() {
    if (this->patchErrorWindow.size() < this->perf_window_size) {
        return true;
    }

    if (this->patchErrorWindowSum < this->transErrorWindowSum) {
        return true;
    }

    return false;
}

bool opera::init_data_source(const string& filename) {
    cout << "Initializing data source..." << endl;

    reader = make_unique<ArffReader>();
    if (!reader->setFile(filename)) {
        cout << "Failed to open file: " << filename << endl;
        exit(1);
    }

    return true;
}

bool opera::get_next_instance() {
    if (!reader->hasNextInstance()) {
        return false;
    }

    instance = reader->nextInstance();
    return true;
}

int opera::get_cur_instance_label() {
    return instance->getLabel();
}

void opera::delete_cur_instance() {
    delete instance;
}

double opera::get_full_region_complexity() {
    return this->full_region_complexity;
}

double opera::get_error_region_complexity() {
    return this->error_region_complexity;
}

double opera::get_correct_region_complexity() {
    return this->correct_region_complexity;
}

// class true_error
opera::true_error::true_error(
            int windowSize,
            double delta,
            double convThreshold,
            std::mt19937& mrand) :
            windowSize(windowSize),
            delta(delta),
            convThreshold(convThreshold),
            mrand(mrand) {

    this->sampleSize = 0;
    this->rc = 0;
    this->errorCount = 0;
    this->windowSum = 0;
}

bool opera::true_error::is_stable(int error) {
    double trueError = this->get_true_error(error);
    this->window.push_back(trueError);
    this->windowSum += trueError;

    if (this->window.size() < this->windowSize) {
        return false;
    }

    if (this->window.size() > this->windowSize) {
        double val = this->window.front();
        this->window.pop_front();
        this->windowSum -= val;
    }

    double mean = this->windowSum / this->window.size();
    double numerator = 0;
    for (double err : window) {
        numerator += sqrt(abs(err - mean));
    }

    double convVal = sqrt(numerator / (this->window.size() - 1));
    if (convVal <= this->convThreshold) {
        return true;
    }

    return false;
}

double opera::true_error::get_true_error(int error) {
    this->errorCount += error;
    this->sampleSize++;

    int sigma = -1;
    auto dist = std::uniform_real_distribution<>(0,1);
    if (dist(this->mrand) < 0.5) {
        sigma = 1;
    }
    this->rc += sigma * error;

    double risk = this->errorCount / this->sampleSize;
    this->rc /= this->sampleSize;

    // true error based on Rademacher bound
    double trueErrorBound = risk + 2*this->rc + 3*sqrt(log(2/this->delta) / (2*sampleSize));
    return trueErrorBound;
}