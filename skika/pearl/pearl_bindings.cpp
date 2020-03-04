#include <pybind11/pybind11.h>
#include "pearl.h"

namespace py = pybind11;

PYBIND11_MODULE(pearl, m) {
    m.doc() = "PEARL's implementation in C++"; // module docstring

    py::class_<pearl>(m, "pearl")
        .def(py::init<int,
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
        .def("init_data_source", &pearl::init_data_source)
        .def("get_next_instance", &pearl::get_next_instance)
        .def("get_cur_instance_label", &pearl::get_cur_instance_label)
        .def("delete_cur_instance", &pearl::delete_cur_instance)
        .def("predict", &pearl::predict)
        .def("train", &pearl::train)
        .def("is_state_graph_stable", &pearl::is_state_graph_stable)
        .def("__repr__",
            [](const pearl &p) {
                return "<pearl.pearl has "
                    + std::to_string(p.get_tree_pool_size()) + " trees>";
            }
         );
}
