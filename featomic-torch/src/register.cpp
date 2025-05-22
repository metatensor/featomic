#include <torch/script.h>

#include "featomic/torch.hpp"
using namespace featomic_torch;

TORCH_LIBRARY(featomic, module) {
    // There is no way to access the docstrings from Python, so we don't bother
    // setting them to something useful here.
    const std::string DOCSTRING;

    module.class_<CalculatorOptionsHolder>("CalculatorOptions")
        .def(torch::init())
        .def_readwrite("gradients", &CalculatorOptionsHolder::gradients)
        .def_property("selected_keys",
            &CalculatorOptionsHolder::selected_keys,
            &CalculatorOptionsHolder::set_selected_keys
        )
        .def_property("selected_samples",
            &CalculatorOptionsHolder::selected_samples,
            &CalculatorOptionsHolder::set_selected_samples
        )
        .def_property("selected_properties",
            &CalculatorOptionsHolder::selected_properties,
            &CalculatorOptionsHolder::set_selected_properties
        )
        ;

    module.class_<CalculatorHolder>("CalculatorHolder")
        .def(torch::init<std::string, std::string>(),
            DOCSTRING,
            {torch::arg("name"), torch::arg("parameters")}
        )
        .def_property("name", &CalculatorHolder::name)
        .def_property("parameters", &CalculatorHolder::parameters)
        .def_property("cutoffs", &CalculatorHolder::cutoffs)
        .def("compute", &CalculatorHolder::compute, DOCSTRING, {
            torch::arg("systems"),
            torch::arg("options") = {}
        })
        .def_pickle(
            // __getstate__
            [](const TorchCalculator& self) -> std::tuple<std::string, std::string> {
                return {self->c_name(), self->parameters()};
            },
            // __setstate__
            [](std::tuple<std::string, std::string> state) -> TorchCalculator {
                return c10::make_intrusive<CalculatorHolder>(
                    std::get<0>(state), std::get<1>(state)
                );
            })
        ;

    module.def(
        "register_autograd("
            "__torch__.torch.classes.metatomic.System[] systems,"
            "__torch__.torch.classes.metatensor.TensorMap precomputed,"
            "str[] forward_gradients"
        ") -> __torch__.torch.classes.metatensor.TensorMap",
        register_autograd
    );
}
