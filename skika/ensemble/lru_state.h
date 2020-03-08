#ifndef __LRU_STATE_H__
#define __LRU_STATE_H__

#include <climits>
#include <unordered_map>
#include <list>
#include <vector>
#include <set>
#include <sstream>

using std::string;
using std::vector;
using std::unordered_map;
using std::list;
using std::set;
using std::stringstream;

class lru_state {
    public:
        lru_state(int capacity, int distance_threshold);

        set<int> get_closest_state(set<int> target_pattern,
                                   set<int> ids_to_exclude);

        void enqueue(set<int> pattern);
        string to_string();

    private:
        struct state {
            set<int> pattern;
            int val;
            int freq;
            state(set<int> pattern, int val, int freq) : pattern(pattern),
            val(val), freq(freq) {}
        };

        list<state> queue;
        unordered_map<string, list<state>::iterator> map;
        int capacity;
        int distance_threshold; 

        string pattern_to_key(const set<int>& pattern);
        void update_queue(const set<int>& pattern);
};

#endif
