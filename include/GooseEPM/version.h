/**
 * Version information.
 *
 * @file version.h
 * @copyright Copyright 2022. Marko Popovic, Tom de Geus. All rights reserved.
 * @license This project is released under the MIT License.
 */

#ifndef GOOSEEPM_VERSION_H
#define GOOSEEPM_VERSION_H

#include "config.h"
#include <prrng.h>

/**
 * Current version.
 *
 * Either:
 *
 * -   Configure using CMake at install time. Internally uses::
 *
 *         python -c "from setuptools_scm import get_version; print(get_version())"
 *
 * -   Define externally using::
 *
 *         MYVERSION=`python -c "from setuptools_scm import get_version; print(get_version())"`
 *         -DGOOSEEPM_VERSION="$MYVERSION"
 *
 *     From the root of this project. This is what `setup.py` does.
 *
 * Note that both `CMakeLists.txt` and `setup.py` will construct the version using
 * `setuptools_scm`. Tip: use the environment variable `SETUPTOOLS_SCM_PRETEND_VERSION` to
 * overwrite the automatic version.
 */
#ifndef GOOSEEPM_VERSION
#define GOOSEEPM_VERSION "@PROJECT_VERSION@"
#endif

namespace GooseEPM {

namespace detail {

inline std::string unquote(const std::string& arg)
{
    std::string ret = arg;
    ret.erase(std::remove(ret.begin(), ret.end(), '\"'), ret.end());
    return ret;
}

} // namespace detail

/**
 * Return version string, e.g. `"0.8.0"`
 * @return String.
 */
inline std::string version()
{
    return detail::unquote(std::string(QUOTE(GOOSEEPM_VERSION)));
}

/**
 * Return versions of this library and of all of its dependencies.
 * The output is a list of strings, e.g.::
 *
 *     "gooseepm=0.7.0",
 *     "xtensor=0.20.1"
 *     ...
 *
 * @return List of strings.
 */
inline std::vector<std::string> version_dependencies()
{
    auto ret = prrng::version_dependencies();
    ret.push_back("gooseepm=" + version());
    std::sort(ret.begin(), ret.end(), std::greater<std::string>());
    return ret;
}

} // namespace GooseEPM

#endif
