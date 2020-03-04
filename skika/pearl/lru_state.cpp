#include "lru_state.h"

lru_state::lru_state(int capacity, int distance_threshold)
    : capacity(capacity), distance_threshold(distance_threshold) {}

set<int> lru_state::get_closest_state(set<int> target_pattern,
                                      set<int> ids_to_exclude) {
    int min_edit_distance = INT_MAX;
    int max_freq = 0;
    set<int>* closest_pattern = nullptr;

    // find the smallest edit distance
    for (auto& cur_state : queue) {
        set<int>& cur_pattern = cur_state.pattern;

        int cur_freq = cur_state.freq;
        int cur_edit_distance = 0;

        bool update_flag = true;
        for (auto& id : ids_to_exclude) {
            if (cur_pattern.find(id) != cur_pattern.end()) {
                // tree with drift must be unset
                update_flag = false;
                break;
            }
        }

        if (update_flag) {
            for (auto& id : target_pattern) {
                if (cur_pattern.find(id) == cur_pattern.end()) {
                    cur_edit_distance++;
                }

                if (cur_edit_distance > distance_threshold
                        || cur_edit_distance > min_edit_distance) {
                    update_flag = false;
                    break;
                }
            }
        }

        if (!update_flag) {
            continue;
        }

        if (min_edit_distance == cur_edit_distance && cur_freq < max_freq) {
            continue;
        }

        min_edit_distance = cur_edit_distance;
        max_freq = cur_freq;
        closest_pattern = &cur_pattern;
    }

    if (!closest_pattern) {
        return {};
    }

    return set<int>(*closest_pattern);
}

void lru_state::update_queue(const set<int>& pattern) {
    string key = pattern_to_key(pattern);

    if (map.find(key) == map.end()) {
        queue.emplace_front(pattern, 1, 1);

    } else {
        auto pos = map[key];
        auto res = *pos;
        res.freq++;

        queue.erase(pos);
        queue.push_front(res);
    }

    map[key] = queue.begin();
}

void lru_state::enqueue(set<int> pattern) {
    update_queue(pattern);

    while (queue.size() > this->capacity) {
        set<int> rm_pattern = queue.back().pattern;
        string rm_pattern_str = pattern_to_key(pattern);
        map.erase(rm_pattern_str);

        queue.pop_back();
    }
}

string lru_state::pattern_to_key(const set<int>& pattern) {
    stringstream ss;
    for (auto& i : pattern) {
        ss << i << ",";
    }

    return ss.str();
}

string lru_state::to_string() {
    string str = "";

    for (auto& s : queue) {
        set<int> cur_pattern = s.pattern;
        string freq = std::to_string(s.freq);

        string delim = "";
        for (int i : cur_pattern) {
            str += delim;
            str += std::to_string(i);
            delim = ",";
        }
        str += ":" + freq + "->";
    }

    return str;
}
