#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "opera_wrapper.h"

namespace py = pybind11;

PYBIND11_MODULE(transfer , m) {
    m.doc() = "online_transfer's implementation in C++";

    py::class_<opera_wrapper>(m, "opera_wrapper")
            .def(py::init<
                    int,
                    int,
                    int,
                    double,
                    // transfer learning params
                    int,
                    int,
                    int,
                    double,
                    double,
                    int,
                    int,
                    int,
                    int,
                    bool,
                    bool,
                    bool>())
            .def("init_data_source", &opera_wrapper::init_data_source)
            .def("get_next_instance", &opera_wrapper::get_next_instance)
            .def("get_cur_instance_label", &opera_wrapper::get_cur_instance_label)
            .def("predict", &opera_wrapper::predict)
            .def("train", &opera_wrapper::train)
            .def("switch_classifier", &opera_wrapper::switch_classifier)
            .def("train", &opera_wrapper::train)
            .def("predict", &opera_wrapper::predict)
            .def("get_cur_instance_label", &opera_wrapper::get_cur_instance_label)
            .def("init_data_source", &opera_wrapper::init_data_source)
            .def("get_next_instance", &opera_wrapper::get_next_instance)
            .def("get_full_region_complexity", &opera_wrapper::get_full_region_complexity)
            .def("get_error_region_complexity", &opera_wrapper::get_error_region_complexity)
            .def("get_correct_region_complexity", &opera_wrapper::get_correct_region_complexity);

}
