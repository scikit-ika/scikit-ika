#include <pybind11/pybind11.h>
#include "adaptive_random_forest.h"
#include "pearl.h"
#include "pro_pearl.h"

namespace py = pybind11;

PYBIND11_MODULE(ensemble, m) {
    m.doc() = "Ensemble techniques' implementations in C++"; // module docstring

    py::class_<adaptive_random_forest>(m, "adaptive_random_forest")
        .def(py::init<int,
                      int,
                      int,
                      int,
                      double,
                      double>())
        .def("init_data_source", &adaptive_random_forest::init_data_source)
        .def("get_next_instance", &adaptive_random_forest::get_next_instance)
        .def("get_cur_instance_label", &adaptive_random_forest::get_cur_instance_label)
        .def("delete_cur_instance", &adaptive_random_forest::delete_cur_instance)
        .def("predict", &adaptive_random_forest::predict)
        .def("train", &adaptive_random_forest::train);

    py::class_<pearl, adaptive_random_forest>(m, "pearl")
        .def(py::init<int,
                      int,
                      int,
                      int,
                      int,
                      int,
                      int,
                      int,
                      int,
                      int,
                      double,
                      double,
                      double,
                      double,
                      double,
                      bool,
                      bool>())
        .def("get_candidate_tree_group_size", &pearl::get_candidate_tree_group_size)
        .def("get_tree_pool_size", &pearl::get_tree_pool_size)
        .def("is_state_graph_stable", &pearl::is_state_graph_stable)
        .def("__repr__",
            [](const pearl &p) {
                return "<pearl.pearl has "
                    + std::to_string(p.get_tree_pool_size()) + " trees>";
            }
         );

    py::class_<pro_pearl, pearl>(m, "pro_pearl")
        .def(py::init<int,
                      int,
                      int,
                      int,
                      int,
                      int,
                      int,
                      int,
                      int,
                      int,
                      double,
                      double,
                      double,
                      double,
                      double,
                      int,
                      double,
                      int,
                      double>())
        .def("select_candidate_trees", &pearl::select_candidate_trees)
        .def("has_actual_drift", &pro_pearl::has_actual_drift)
        .def("find_last_actual_drift_point", &pro_pearl::find_last_actual_drift_point)
        .def("train", &pro_pearl::train)
        .def("adapt_state", &pro_pearl::adapt_state)
        .def("adapt_state_with_proactivity", &pro_pearl::adapt_state_with_proactivity)
        .def("set_expected_drift_prob", &pro_pearl::set_expected_drift_prob)
        .def("get_stable_tree_indices", &pro_pearl::get_stable_tree_indices);
}
