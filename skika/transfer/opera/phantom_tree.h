#ifndef ONLINE_TRANSFER_PHANTOM_TREE_H
#define ONLINE_TRANSFER_PHANTOM_TREE_H

#include <streamDM/learners/Classifiers/Trees/HoeffdingTree.h>
#include <streamDM/learners/Classifiers/Trees/HTNode.h>
#include <algorithm>

class phantom_tree : public HT::HoeffdingTree {

public:
    class phantom_node;

    phantom_tree(
            int num_phantom_branches,
            double squashing_delta,
            int split_range,
            std::mt19937& mrand);
    double get_construction_complexity(vector<Instance*>& instance_store);
    deque<shared_ptr<phantom_node>> growPhantomBranches();
    shared_ptr<phantom_node> growPhantomBranch(shared_ptr<phantom_node> node);
    void phantomSplit(shared_ptr<phantom_node> node);
    double calcPhantomInfoGain(shared_ptr<phantom_node> child);
    int getWeightedRandomPhantomNodeIdx(vector<shared_ptr<phantom_node>>& phantomChildren);
    deque<shared_ptr<phantom_node>> initPhantomRootParents();
    void addInstancesToLeaves(deque<Instance*>& instanceStore);
    void filter(HT::Node* node, Instance* inst);
    void printPhantomBranches(deque<phantom_node>& phantomLeaves);

    double avgPhantomBranchDepth = -1;
    int num_phantom_branches = 7;
    double squashing_delta = 7;
    int split_range = 10;
    std::mt19937 mrand;
    deque<Instance*> instance_store;


    class phantom_node : public HT::ActiveLearningNode {
        public:

            int depth;
            double foil_info_gain;
            // public boolean isPhantomLeaf;
            string branchPrefix = "";

            vector<HT::InstanceConditionalTest*> splitTests;
            vector<shared_ptr<phantom_node>> splitChildrenPairs;
            deque<Instance*> instanceStore;

            phantom_node(const vector<double>& initialClassObservations,
                         mt19937& mrand,
                         int depth,
                         string branchPrefix);

            void passInstanceToChild(
                Instance* inst,
                HoeffdingTree* ht,
                HT::InstanceConditionalTest* splitTest,
                vector<shared_ptr<phantom_node>>& children);

            // vector<HT::AttributeSplitSuggestion*> getAllSplitSuggestions(HT::SplitCriterion* criterion);

            // string phantom_tree::phantom_node::toString() override {
            //     return this->branchPrefix;
            // }

    };

};


#endif //ONLINE_TRANSFER_PHANTOM_TREE_H