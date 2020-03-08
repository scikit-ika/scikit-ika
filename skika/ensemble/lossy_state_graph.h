#ifndef __LOSSY_STATE_GRAPH_H__
#define __LOSSY_STATE_GRAPH_H__

#include <iostream>
#include <vector>
#include <map>
#include <memory>
#include <queue>
#include <sstream>
#include <string>
#include <random>

using std::cout;
using std::endl;
using std::to_string;
using std::unique_ptr;
using std::shared_ptr;
using std::make_unique;
using std::map;
using std::vector;
using std::queue;
using std::string;
using std::stringstream;

class lossy_state_graph {
    public:

        lossy_state_graph(int capacity, int window_size, std::mt19937 mrand);
        int get_next_tree_id(int src);
        bool update(int warning_tree_count);

        void try_remove_node(int key);
        void add_node(int key);
        void add_edge(int src, int dest);

        void set_is_stable(bool is_stable_);
        bool get_is_stable();
        string to_string();

    private:

        struct node_t {
            int indegree;
            int total_weight;
            map<int, int> neighbors; // <tree_id, freq>
        };

        vector<unique_ptr<node_t>> graph;
        int capacity;
        int window_size;
        std::mt19937 mrand;

        int drifted_tree_counter = 0;
        bool is_stable;
};

class state_graph_switch {
    public:
        state_graph_switch(shared_ptr<lossy_state_graph> state_graph,
                           int window_size,
                           double reuse_rate);

        void update_reuse_count(int num_reused_trees);
        void update_switch();

    private:
        int window_size = 0;
        int reused_tree_count = 0;
        int total_tree_count = 0;
        double reuse_rate = 1.0;

        queue<int> window;
        shared_ptr<lossy_state_graph> state_graph;
};

#endif
