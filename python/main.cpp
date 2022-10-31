/**
 * @file
 * @copyright Copyright 2020. Tom de Geus. All rights reserved.
 * @license This project is released under the GNU Public License (MIT).
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#define FORCE_IMPORT_ARRAY
#include <xtensor-python/pyarray.hpp>
#include <xtensor-python/pytensor.hpp>
#include <xtensor-python/xtensor_python_config.hpp>

#define GOOSEEPM_USE_XTENSOR_PYTHON
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
}
