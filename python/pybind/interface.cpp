#include <pybind11/pybind11.h>

#include "rsearch_type.h"
#include "gallery/rsearch_gallery.h"
#include "probe/rsearch_probe.h"

namespace py = pybind11;

PYBIND11_MODULE(rsearch, m){
    m.doc() = ""

    m.def("create_probe", &create_prbe<int8_t>
    py::class_<gallery>(m, "gallery")
                    .def(py::init<>())
                    .def("init", &gallery::init)
                    .def("add", &gallery::add)
                    .def("add_with_uids", &gallery::add_with_uids)
                    .def("change_by_uids", &gallery::change_by_uids)
                    .def("query_by_uids", &gallery::query_by_uids)
                    .def("reset", &gallery::reset)
                    .def("store_data", &gallery::store)
                    .def("")
    );
    


}