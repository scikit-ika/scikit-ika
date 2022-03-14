#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "trans_tree_wrapper.h"

namespace py = pybind11;

PYBIND11_MODULE(aotradaboost, m) {
    m.doc() = "online_transfer's implementation in C++";

    py::class_<trans_tree_wrapper>(m, "trans_tree_wrapper")
            .def(py::init<
                    int,
                    int,
                    int,
                    double,
                    double,
                    // transfer learning params
                    int,
                    int,
                    int,
                    int,
                    int,
                    double,
                    double,
                    double,
                    string,
                    int,
                    bool>())
            .def("init_data_source", &trans_tree_wrapper::init_data_source)
            .def("get_next_instance", &trans_tree_wrapper::get_next_instance)
            .def("get_cur_instance_label", &trans_tree_wrapper::get_cur_instance_label)
            .def("predict", &trans_tree_wrapper::predict)
            .def("train", &trans_tree_wrapper::train)
            .def("switch_classifier", &trans_tree_wrapper::switch_classifier)
            .def("train", &trans_tree_wrapper::train)
            .def("predict", &trans_tree_wrapper::predict)
            .def("get_cur_instance_label", &trans_tree_wrapper::get_cur_instance_label)
            .def("init_data_source", &trans_tree_wrapper::init_data_source)
            .def("get_next_instance", &trans_tree_wrapper::get_next_instance)
            .def("get_transferred_tree_group_size", &trans_tree_wrapper::get_transferred_tree_group_size)
            .def("get_tree_pool_size", &trans_tree_wrapper::get_tree_pool_size);

}
