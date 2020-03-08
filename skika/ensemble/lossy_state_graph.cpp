#include "lossy_state_graph.h" 

lossy_state_graph::lossy_state_graph(int capacity,
                                     int window_size,
                                     std::mt19937 mrand)
        : capacity(capacity), 
          window_size(window_size),
          mrand(mrand) {

    is_stable = false;
    graph = vector<unique_ptr<node_t>>(capacity);
}

int lossy_state_graph::get_next_tree_id(int src) {
    if (!graph[src] || graph[src]->total_weight == 0) {
        return -1;
    }

    std::uniform_int_distribution<> uniform_distr(0, graph[src]->total_weight);
    int r = uniform_distr(mrand);
    int sum = 0;

    // weighted selection
    for (auto nei : graph[src]->neighbors) {
        sum += nei.second;
        if (r < sum) {
            graph[src]->neighbors[nei.first]++;
            graph[src]->total_weight++;
            return nei.first;
        }
    }

    return -1;
}

bool lossy_state_graph::update(int warning_tree_count) {
    drifted_tree_counter += warning_tree_count;

    if (drifted_tree_counter < window_size) {
        return false;
    }

    drifted_tree_counter -= window_size;

    // lossy count
    for (int i = 0; i < graph.size(); i++) {
        if (!graph[i]) {
            continue;
        }

        vector<int> keys_to_remove;

        for (auto& nei : graph[i]->neighbors) {
            // decrement freq by 1
            graph[i]->total_weight--;
            nei.second--; // freq

            if (nei.second == 0) {
                // remove edge
                graph[nei.first]->indegree--;
                try_remove_node(nei.first);

                keys_to_remove.push_back(nei.first);
            }
        }

        for (auto& key : keys_to_remove) {
            graph[i]->neighbors.erase(key);
        }

        try_remove_node(i);
    }

    return true;
}

void lossy_state_graph::try_remove_node(int key) {
    if (graph[key]->indegree == 0 && graph[key]->neighbors.size() == 0) {
        graph[key].reset();
    }
}

void lossy_state_graph::add_node(int key) {
    if (key >= capacity) {
        cout << "id exceeded graph capacity" << endl;
        return;
    }

    graph[key] = make_unique<node_t>();
}

void lossy_state_graph::add_edge(int src, int dest) {
    if (!graph[src]) {
        add_node(src);
    }

    if (!graph[dest]) {
        add_node(dest);
    }

    graph[src]->total_weight++;

    if (graph[src]->neighbors.find(dest) == graph[src]->neighbors.end()) {
        graph[src]->neighbors[dest] = 0;
        graph[dest]->indegree++;
    }

    graph[src]->neighbors[dest]++;
}

void lossy_state_graph::set_is_stable(bool is_stable_) {
    is_stable = is_stable_;
}

bool lossy_state_graph::get_is_stable() {
    return is_stable;
}

string lossy_state_graph::to_string() {
    stringstream ss;
    for (int i = 0; i < graph.size(); i++) {
        ss << i;
        if (!graph[i]) {
            ss << " {}" << endl;
            continue;
        }

        ss << " w:" << std::to_string(graph[i]->total_weight) << " {";
        for (auto& nei : graph[i]->neighbors) {
            ss << std::to_string(nei.first) << ":" << std::to_string(nei.second) << " ";
        }
        ss << "}" << endl;
    }

    return ss.str();
}


// graph switch
state_graph_switch::state_graph_switch(shared_ptr<lossy_state_graph> state_graph,
                                       int window_size,
                                       double reuse_rate)
        : state_graph(state_graph),
          window_size(window_size),
          reuse_rate(reuse_rate) { }

void state_graph_switch::update_reuse_count(int num_reused_trees) {
    reused_tree_count += num_reused_trees;
    total_tree_count++;

    if (window_size <= 0) {
        return;
    }

    if (window.size() >= window_size) {
        reused_tree_count -= window.front();
        window.pop();
    }

    window.push(num_reused_trees);
}

void state_graph_switch::update_switch() {
    double cur_reuse_rate = 0;
    if (window_size <= 0) {
        cur_reuse_rate = (double) reused_tree_count / total_tree_count;
    } else {
        cur_reuse_rate = (double) reused_tree_count / window_size;
    }

    // cout << "reused_tree_count: " << to_string(reused_tree_count)  << endl;
    // cout << "total_tree_count: " << to_string(total_tree_count)  << endl;
    // cout << "cur_reuse_rate: " << to_string(cur_reuse_rate)  << endl;

    if (cur_reuse_rate >= reuse_rate) {
        state_graph->set_is_stable(true);
    } else {
        state_graph->set_is_stable(false);
    }
}
