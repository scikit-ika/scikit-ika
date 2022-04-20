#include "phantom_tree.h"

phantom_tree::phantom_tree(
        int num_phantom_branches,
        double squashing_delta,
        int split_range,
        std::mt19937& mrand) :
        num_phantom_branches(num_phantom_branches),
        squashing_delta(squashing_delta),
        split_range(split_range),
        mrand(mrand) {
    this->params.binarySplits = true;
    this->params.noPrePrune = true;
    // this->params.numericEstimator = "-n 2";
    this->mrand = mrand;
}

double phantom_tree::get_construction_complexity(vector<Instance*>& instances) {
    cout << "get_construction_complexity..." << endl;

    for (Instance* inst : instances) {
        HoeffdingTree::train(*inst);
        this->instance_store.push_back(inst);
    }
    cout << "original tree" <<  this->printTree() << endl;

    deque<shared_ptr<phantom_node>> phantomLeaves = growPhantomBranches();
    double depth_sum = 0.0;
    for (shared_ptr<phantom_node> phantomLeaf : phantomLeaves) {
        depth_sum += phantomLeaf->depth;
    }

    if (phantomLeaves.size() == 0) {
        return 0;
    }

    return depth_sum / phantomLeaves.size();
}

deque<shared_ptr<phantom_tree::phantom_node>> phantom_tree::growPhantomBranches() {
    // init candidate phantom branches
    cout << "init candidate phantom branches" << endl;
    addInstancesToLeaves(this->instance_store);
    deque<shared_ptr<phantom_node>> phantomRootParents = initPhantomRootParents();
    if (phantomRootParents.size() == 0) {
        cout << "No phantom root parent constructed." << endl;
        exit(1);
    }

    cout << "first level phantom splits" << endl;
    // perform first level phantom splits to find phantom roots
    vector<shared_ptr<phantom_node>> phantomRoots;
    for (shared_ptr<phantom_node> parent : phantomRootParents) {
        phantomSplit(parent);
        phantomRoots.insert(
                phantomRoots.end(),
                parent->splitChildrenPairs.begin(),
                parent->splitChildrenPairs.end()); // concat two vectors
    }

    deque<shared_ptr<phantom_node>> phantomLeaves;

    // TODO if the first level split produces zero children, the complexity equals 1.
    if (phantomRoots.size() == 0) {
        this->avgPhantomBranchDepth = 1;
        return phantomLeaves;
    }

    double sum_depth = 0.0;
    for (int i = 0; i < this->num_phantom_branches; i++) {
        cout << "Start constructing branch " << i << endl;
        int nodeIdx = getWeightedRandomPhantomNodeIdx(phantomRoots);
        cout << "obtained nodeIdx" << endl;
        if (nodeIdx == -1 && nodeIdx >= phantomRoots.size()) {
            cout << "getWeightedRandomPhantomNodeIdx out of range" << endl;
            exit(1);
        }
        shared_ptr<phantom_node> phantomRoot = phantomRoots[nodeIdx];

        // shared_ptr<stringstream> branchStringBuilder;
        // (*branchStringBuilder) << phantomRoot->branchPrefix;
        cout << "#";

        cout << "Phantom Branch " << i << ": " << flush;
        shared_ptr<phantom_node> phantomLeaf = growPhantomBranch(phantomRoot);
        phantomLeaves.push_back(phantomLeaf);
        sum_depth = sum_depth + phantomLeaf->depth;

        // cout << "Phantom Branch " + i << ": " << branchStringBuilder->str() << endl;
        cout << endl;
    }

    cout << "avg=" << sum_depth / phantomLeaves.size() << flush << endl;

    this->avgPhantomBranchDepth = sum_depth / phantomLeaves.size();

    return phantomLeaves;
}

shared_ptr<phantom_tree::phantom_node> phantom_tree::growPhantomBranch(shared_ptr<phantom_node> node) {
    if  (node == nullptr) {
        cout << "growPhantomBranch node is null" << endl;
    }

    if (node->isPure()) {
        return node;
    }

    // TODO cache leaf node info
    // split if phantom children do not exist
    if (node->splitChildrenPairs.size() == 0) {
        phantomSplit(node);
    }

    int childIdx = getWeightedRandomPhantomNodeIdx(node->splitChildrenPairs);
    if (childIdx == -1) {
        cout << "getWeightedRandomPhantomChildIdx returns -1" << endl;
        return node;
    }
    shared_ptr<phantom_node> selectedPhantomChild = node->splitChildrenPairs[childIdx];
    HT::InstanceConditionalTest* condition = node->splitTests[childIdx];
    if (dynamic_cast<HT::NominalAttributeBinaryTest*>(condition)) {
        cout << condition->getAttIndex() << ",";
        // *branchStringBuilder << condition->getAttIndex() << ",";
    } else if (dynamic_cast<HT::NumericAttributeBinaryTest*>(condition)) {
        cout << condition->getAttIndex() << ",";
        // *branchStringBuilder << condition->getAttIndex() << ",";
    } else if (condition == nullptr) {
        cout << "splitTest is null." << endl;
        exit(1);
    } else {
        cout << "Multiway test is not supported." << endl;
        exit(1);
    }
    // System.out.println(condition.getAttributeIndex() + ":" + condition.getAttributeValue());

    return growPhantomBranch(selectedPhantomChild);
}

bool compare_suggestion(HT::AttributeSplitSuggestion* v1,
                        HT::AttributeSplitSuggestion* v2) {
    return v1->merit < v2->merit;
}

void phantom_tree::phantomSplit(shared_ptr<phantom_node> node) {
    HT::SplitCriterion* splitCriterion = new HT::InfoGainSplitCriterion();
    list<HT::AttributeSplitSuggestion*>* allSplitSuggestions = node->getBestSplitSuggestions(splitCriterion, this);
    if (allSplitSuggestions->size() == 0) {
        return;
    }
    allSplitSuggestions->sort(compare_suggestion);
    allSplitSuggestions->reverse();
    allSplitSuggestions->resize(std::min((size_t) this->split_range, allSplitSuggestions->size()));

    auto iter = (*allSplitSuggestions).begin();
    for (; iter != (*allSplitSuggestions).end(); iter++) {
        if ((*iter)->splitTest == nullptr) {
            cout << "PhantomSplit splitTest is null." << endl;
        }

        HT::InstanceConditionalTest* curSplitTest = (*iter)->splitTest;
        vector<shared_ptr<phantom_node>> newChildren;
        bool isUsedAttribute = false;
        for (int i = 0; i < (*iter)->numSplits(); i++) {
            vector<double>* resultingClassDistribution = (*iter)->resultingClassDistributionFromSplit(i);
            bool shouldSplit = false;
            for (int j = 0; j < resultingClassDistribution->size(); j++) {
                if ((*resultingClassDistribution)[j] != node->getObservedClassDistribution()[j]) {
                    shouldSplit = true;
                    break;
                }
            }
            if (!shouldSplit) {
                isUsedAttribute = true;
                break;
            }

        }

        if (!isUsedAttribute) {
            for (int i = 0; i < (*iter)->numSplits(); i++) {
                shared_ptr<phantom_node> newChild = make_shared<phantom_node>(
                        vector<double>(0),
                        this->mrand,
                        node->depth + 1,
                        node->branchPrefix);

                newChildren.push_back(newChild);

                node->splitTests.push_back(curSplitTest);
                node->splitChildrenPairs.push_back(newChild);
            }

            // TODO only train the selected phantom children?
            // for each splitTest, pass instances & train
            for (Instance* inst : node->instanceStore) {
                node->passInstanceToChild(inst, this, curSplitTest, newChildren);
            }

            // compute foil information gain for weighted selection
            for (shared_ptr<phantom_node> phantomChild : newChildren) {
                phantomChild->foil_info_gain = calcPhantomInfoGain(phantomChild);
            }
        }
    }

    // cout << "phantom splits end" << endl;
    cout << flush;
}

double phantom_tree::calcPhantomInfoGain(shared_ptr<phantom_node> child) {
    double total = child->instanceStore.size();
    if (total == 0) {
        cout << "empty instanceStore" << endl;
        return -1;
    }

    double child_num_positive = 0;
    for (Instance* inst : child->instanceStore) {
        int trueClass = (int) inst->getLabel();
        vector<double> child_predictions = child->getClassVotes(inst, this);
        int childPrediction = 0;
        double max_val = child_predictions[0];
        // Find class label with the highest probability
        for (int i = 1; i < inst->getNumberClasses(); i++) {
            if (max_val < child_predictions[i]) {
                max_val = child_predictions[i];
                childPrediction = i;
            }
        }

        if (childPrediction == trueClass) {
            child_num_positive++;
        }
    }

    double phantom_factor = - log(1 - child_num_positive / total) / log(2);
    if (phantom_factor > this->squashing_delta) {
        phantom_factor = this->squashing_delta;
    }
    return child_num_positive * phantom_factor;
}

int phantom_tree::getWeightedRandomPhantomNodeIdx(vector<shared_ptr<phantom_node>>& phantomChildren) {
    double sum = 0;
    int invalid_child_count = 0;
    for (shared_ptr<phantom_node> child : phantomChildren) {
        if (child->foil_info_gain <= 0) {
            invalid_child_count++;
            continue;
        }
        sum += child->foil_info_gain;
    }

    if (invalid_child_count == phantomChildren.size()) {
        return -1;
    }

    std::uniform_real_distribution<double> random_double(0, 1);
    double rand = random_double(mrand);

    double partial_sum = 0;
    for (int i = 0; i < phantomChildren.size(); i++) {
        shared_ptr<phantom_node> child = phantomChildren[i];
        if (child->foil_info_gain <= 0) {
            continue;
        }
        partial_sum += (child->foil_info_gain / sum);
        if (partial_sum > rand) {
            return i;
        }
    }

    return -1;
}

deque<shared_ptr<phantom_tree::phantom_node>> phantom_tree::initPhantomRootParents() {
    cout << "initPhantomRootParents()" << endl;

    deque<HT::Node*> nodes;
    deque<shared_ptr<phantom_node>> phantomRoots;
    deque<int> depths;
    deque<shared_ptr<stringstream>> branchStringBuilders;
    nodes.push_back(this->treeRoot);
    depths.push_back(1);
    branchStringBuilders.push_back(make_shared<stringstream>("#"));

    while (nodes.size() != 0) {
        HT::Node* curNode = nodes.front();
        nodes.pop_front();
        shared_ptr<stringstream> branchStringBuilder = branchStringBuilders.front();
        branchStringBuilders.pop_front();
        int depth = depths.front();
        depths.pop_front();

        if (dynamic_cast<HT::LearningNode*>(curNode)) {
            *branchStringBuilder << "#";
            shared_ptr<phantom_tree::phantom_node> root = make_shared<phantom_node>(
                    vector<double>(0),
                    mrand,
                    depth,
                    branchStringBuilder->str());
            for (Instance* inst : curNode->instanceStore) { // TODO: add new data struct to Node?
                root->learnFromInstance(inst, this);
                root->instanceStore.push_back(inst);
            }
            phantomRoots.push_back(root);

        } else {
            *branchStringBuilder << ",";

            HT::SplitNode* splitNode = dynamic_cast<HT::SplitNode*>(curNode);
            HT::InstanceConditionalTest* condition = splitNode->splitTest;

            if (dynamic_cast<HT::NominalAttributeBinaryTest*>(condition)) {
                *branchStringBuilder << condition->getAttIndex() << ",";
            } else if (dynamic_cast<HT::NumericAttributeBinaryTest*>(condition)) {
                *branchStringBuilder << condition->getAttIndex() << ",";
            } else {
                cout << "Multiway test is not supported." << endl;
                exit(1);
            }

            for (int j = 0; j < splitNode->numChildren(); j++) {
                nodes.push_back(splitNode->getChild(j));
                branchStringBuilders.push_back(make_shared<stringstream>(branchStringBuilder->str()));
                depths.push_back(depth + 1);
            }
        }
    }

    cout << "initPhantomRootParents()=" << phantomRoots.size() << endl;
    return phantomRoots;
}

void phantom_tree::addInstancesToLeaves(deque<Instance*>& instanceStore) {
    for (Instance* inst : instanceStore) {
        filter(this->treeRoot, inst);
    }
}

void phantom_tree::filter(HT::Node* node, Instance* inst) {
    if (dynamic_cast<HT::LearningNode*>(node)) {
        node->instanceStore.push_back(inst);
        return;
    }
    int childIndex = ((HT::SplitNode*) node)->instanceChildIndex(inst);
    if (childIndex < 0) return;
    // TODO check dynamic cast vs dynamic ptr cast
    HT::Node* child = (dynamic_cast<HT::SplitNode*>(node))->getChild(childIndex);
    filter(child, inst);
}

void phantom_tree::printPhantomBranches(deque<phantom_node>& phantomLeaves) {
    cout << "Phantom Nodes:" << endl;
    for (phantom_node pl: phantomLeaves) {
        cout << "Depth= " << pl.depth << endl;
        cout << pl.branchPrefix << endl;
    }
}


// class phantom_node
phantom_tree::phantom_node::phantom_node(
        const vector<double>& initialClassObservations,
        mt19937& mrand,
        int depth,
        string branchPrefix) : ActiveLearningNode(initialClassObservations, mrand) {
    // this->super(new double[0]);
    this->depth = depth;
    this->branchPrefix = branchPrefix;
    this->foil_info_gain = -1;

    // this.splitTests = new AutoExpandVector<>();
    // this.splitChildrenPairs = new AutoExpandVector<>();
    // this.instanceStore = new ArrayDeque<>();
}

void phantom_tree::phantom_node::passInstanceToChild(
        Instance* inst,
        HoeffdingTree* ht,
        HT::InstanceConditionalTest* splitTest,
        vector<shared_ptr<phantom_node>>& children) {
    int childIndex = splitTest->branchForInstance(inst);
    if (childIndex < 0) {
        cout << "child index less than 0" << endl;
        exit(1);
    }
    shared_ptr<phantom_node> child = children[childIndex];
    if (child == nullptr) {
        cout << "passInstanceTodChild child is null" << endl;
        exit(1);
    }
    child->instanceStore.push_back(inst);
    child->learnFromInstance(inst, ht);
}


// TODO replace getAll with getBest
// list<HT::AttributeSplitSuggestion*>* phantom_tree::phantom_node::getBestSplitSuggestions(HT::SplitCriterion* criterion) {
//     return this->super(criterion, this);
// }