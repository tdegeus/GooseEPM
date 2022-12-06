/**
 * @file
 * @copyright Copyright 2022. Tom de Geus. All rights reserved.
 * @license This project is released under the MIT License.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#define FORCE_IMPORT_ARRAY
#include <xtensor-python/pytensor.hpp>
#include <xtensor-python/xtensor_python_config.hpp>

#define GOOSEEPM_USE_XTENSOR_PYTHON
#include <GooseEPM/System.h>
#include <GooseEPM/version.h>

namespace py = pybind11;

/**
 * Overrides the `__name__` of a module.
 * Classes defined by pybind11 use the `__name__` of the module as of the time they are defined,
 * which affects the `__repr__` of the class type objects.
 */
class ScopedModuleNameOverride {
public:
    explicit ScopedModuleNameOverride(py::module m, std::string name) : module_(std::move(m))
    {
        original_name_ = module_.attr("__name__");
        module_.attr("__name__") = name;
    }
    ~ScopedModuleNameOverride()
    {
        module_.attr("__name__") = original_name_;
    }

private:
    py::module module_;
    py::object original_name_;
};

PYBIND11_MODULE(_GooseEPM, mod)
{
    // Ensure members to display as `GooseEPM.X` (not `GooseEPM._GooseEPM.X`)
    ScopedModuleNameOverride name_override(mod, "GooseEPM");

    xt::import_numpy();

    mod.doc() = "Friction model based on GooseFEM and GooseEPM";

    namespace M = GooseEPM;

    mod.def("version", &M::version, "Return version string.");

    {
        py::class_<M::SystemAthermal> cls(mod, "SystemAthermal");

        cls.def(
            py::init<
                const M::array_type::tensor<double, 2>&,
                const M::array_type::tensor<ptrdiff_t, 1>&,
                const M::array_type::tensor<ptrdiff_t, 1>&,
                const M::array_type::tensor<double, 2>&,
                const M::array_type::tensor<double, 2>&,
                uint64_t,
                double,
                double,
                double,
                bool,
                bool>(),
            "System",
            py::arg("propagator"),
            py::arg("distances_rows"),
            py::arg("distances_cols"),
            py::arg("sigmay_mean"),
            py::arg("sigmay_std"),
            py::arg("seed"),
            py::arg("failure_rate") = 1,
            py::arg("alpha") = 1.5,
            py::arg("sigmabar") = 0,
            py::arg("init_random_stress") = true,
            py::arg("init_relax") = true);

        cls.def("__copy__", [](const M::SystemAthermal& self) { return M::SystemAthermal(self); });

        cls.def("__deepcopy__", [](const M::SystemAthermal& self, py::dict) {
            return M::SystemAthermal(self);
        });

        cls.def_property_readonly("shape", &M::SystemAthermal::shape, "Shape");

        cls.def_property("t", &M::SystemAthermal::t, &M::SystemAthermal::set_t, "Current time");

        cls.def_property(
            "state",
            &M::SystemAthermal::state,
            &M::SystemAthermal::set_state,
            "State of the random number generator");

        cls.def_property(
            "epsp", &M::SystemAthermal::epsp, &M::SystemAthermal::set_epsp, "Plastic strain");

        cls.def_property(
            "sigmay", &M::SystemAthermal::sigmay, &M::SystemAthermal::set_sigmay, "Yield stress");

        cls.def_property(
            "sigma", &M::SystemAthermal::sigma, &M::SystemAthermal::set_sigma, "Stress");

        cls.def_property(
            "sigmabar",
            &M::SystemAthermal::sigmabar,
            &M::SystemAthermal::set_sigmabar,
            "Average (prescribed) stress");

        cls.def(
            "initSigmaFast",
            &M::SystemAthermal::initSigmaFast,
            "Randomise stress field",
            py::arg("sigma_std"),
            py::arg("delta_t"));

        cls.def(
            "initSigmaPropogator",
            &M::SystemAthermal::initSigmaPropogator,
            "Randomise stress field",
            py::arg("sigma_std"));

        cls.def(
            "makeAthermalFailureStep",
            &M::SystemAthermal::makeAthermalFailureStep,
            "Make an normal failure step");

        cls.def(
            "makeWeakestFailureStep",
            &M::SystemAthermal::makeWeakestFailureStep,
            "Fail the weakest block");

        cls.def(
            "spatialParticleFailure",
            &M::SystemAthermal::spatialParticleFailure,
            "Fail a block and apply the propagator",
            py::arg("idx"));

        cls.def(
            "shiftImposedShear",
            &M::SystemAthermal::shiftImposedShear,
            "Increment imposed shear",
            py::arg("direction") = 1);

        cls.def(
            "relaxAthermal",
            &M::SystemAthermal::relaxAthermal,
            "Event-driven step",
            py::arg("max_steps") = 1000000,
            py::arg("max_steps_is_error") = true);

        cls.def(
            "relaxWeakest",
            &M::SystemAthermal::relaxWeakest,
            "Event-driven step",
            py::arg("max_steps") = 1000000,
            py::arg("max_steps_is_error") = true);

        cls.def("__repr__", [](const M::SystemAthermal&) { return "<GooseEPM.SystemAthermal>"; });
    }

    {
        py::class_<M::SystemThermal, M::SystemAthermal> cls(mod, "SystemThermal");

        cls.def(
            py::init<
                const M::array_type::tensor<double, 2>&,
                const M::array_type::tensor<ptrdiff_t, 1>&,
                const M::array_type::tensor<ptrdiff_t, 1>&,
                const M::array_type::tensor<double, 2>&,
                const M::array_type::tensor<double, 2>&,
                uint64_t,
                double,
                double,
                double,
                double,
                bool,
                bool>(),
            "System",
            py::arg("propagator"),
            py::arg("distances_rows"),
            py::arg("distances_cols"),
            py::arg("sigmay_mean"),
            py::arg("sigmay_std"),
            py::arg("seed"),
            py::arg("temperature"),
            py::arg("failure_rate") = 1,
            py::arg("alpha") = 1.5,
            py::arg("sigmabar") = 0,
            py::arg("init_random_stress") = true,
            py::arg("init_relax") = true);

        cls.def("__copy__", [](const M::SystemThermal& self) { return M::SystemThermal(self); });

        cls.def("__deepcopy__", [](const M::SystemThermal& self, py::dict) {
            return M::SystemThermal(self);
        });

        cls.def_property(
            "temperature",
            &M::SystemThermal::temperature,
            &M::SystemThermal::set_temperature,
            "Applied temperature");

        cls.def(
            "makeThermalFailureStep",
            &M::SystemThermal::makeThermalFailureStep,
            "Make a thermal failure step");

        cls.def(
            "makeThermalFailureSteps",
            &M::SystemThermal::makeThermalFailureSteps,
            "Make `n` failure steps",
            py::arg("n"));

        cls.def("__repr__", [](const M::SystemThermal&) { return "<GooseEPM.SystemThermal>"; });
    }

    {
        py::module submod = mod.def_submodule("detail", "detail");
        namespace SM = GooseEPM::detail;

        submod.def(
            "create_distance_lookup",
            &SM::create_distance_lookup<M::array_type::tensor<ptrdiff_t, 1>>,
            "Create a distance lookup.",
            py::arg("distance"));
    }
}
