/**
 * @file
 * @copyright Copyright 2022. Marko Popovic, Tom de Geus. All rights reserved.
 * @license This project is released under the GNU Public License (MIT).
 */

#ifndef GOOSEEPM_CONFIG_H
#define GOOSEEPM_CONFIG_H

/**
 * \cond
 */
#define Q(x) #x
#define QUOTE(x) Q(x)

#define GOOSEEPM_WARNING_IMPL(message, file, line, function) \
    std::cout << std::string(file) + ":" + std::to_string(line) + " (" + std::string(function) + \
                     ")" + ": " message ") \n\t";

#define GOOSEEPM_ASSERT_IMPL(expr, assertion, file, line, function) \
    if (!(expr)) { \
        throw assertion( \
            std::string(file) + ":" + std::to_string(line) + " (" + std::string(function) + ")" + \
            ": assertion failed (" #expr ") \n\t"); \
    }

/**
 * \endcond
 */

/**
 * All assertions are implementation as::
 *
 *     GOOSEEPM_ASSERT(...)
 *
 * They can be enabled by::
 *
 *     #define GOOSEEPM_ENABLE_ASSERT
 *
 * (before including GooseEPM).
 * The advantage is that:
 *
 * -   File and line-number are displayed if the assertion fails.
 * -   GooseEPM's assertions can be enabled/disabled independently from those of other libraries.
 *
 * \throw std::runtime_error
 */
#ifdef GOOSEEPM_ENABLE_ASSERT
#define GOOSEEPM_ASSERT(expr, assertion) \
    GOOSEEPM_ASSERT_IMPL(expr, assertion, __FILE__, __LINE__, __FUNCTION__)
#else
#define GOOSEEPM_ASSERT(expr, assertion)
#endif

/**
 * Assertion that cannot be switched off. Implement assertion by::
 *
 *     GOOSEEPM_REQUIRE(...)
 *
 * \throw std::runtime_error
 */
#define GOOSEEPM_REQUIRE(expr, assertion) \
    GOOSEEPM_ASSERT_IMPL(expr, assertion, __FILE__, __LINE__, __FUNCTION__)

/**
 * Assertion that concerns temporary implementation limitations.
 * Implement assertion by::
 *
 *     GOOSEEPM_WIP_ASSERT(...)
 *
 * \throw std::runtime_error
 */
#define GOOSEEPM_WIP_ASSERT(expr, assertion) \
    GOOSEEPM_ASSERT_IMPL(expr, assertion, __FILE__, __LINE__, __FUNCTION__)

/**
 * All warnings are implemented as::
 *
 *     GOOSEEPM_WARNING(...)
 *
 * They can be disabled by::
 *
 *     #define GOOSEEPM_DISABLE_WARNING
 */
#ifdef GOOSEEPM_DISABLE_WARNING
#define GOOSEEPM_WARNING(message)
#else
#define GOOSEEPM_WARNING(message) GOOSEEPM_WARNING_IMPL(message, __FILE__, __LINE__, __FUNCTION__)
#endif

/**
 * All warnings specific to the Python API are implemented as::
 *
 *     GOOSEEPM_WARNING_PYTHON(...)
 *
 * They can be enabled by::
 *
 *     #define GOOSEEPM_ENABLE_WARNING_PYTHON
 */
#ifdef GOOSEEPM_ENABLE_WARNING_PYTHON
#define GOOSEEPM_WARNING_PYTHON(message) \
    GOOSEEPM_WARNING_IMPL(message, __FILE__, __LINE__, __FUNCTION__)
#else
#define GOOSEEPM_WARNING_PYTHON(message)
#endif

/**
 * Toolbox to perform finite element computations.
 */
namespace GooseEPM {

/**
 * Container type.
 * By default `array_type::tensor` is used. Otherwise:
 *
 * -   `#define GOOSEEPM_USE_XTENSOR_PYTHON` to use `xt::pytensor`
 */
namespace array_type {

#ifdef GOOSEEPM_USE_XTENSOR_PYTHON

/**
 * Fixed (static) rank array.
 */
template <typename T, size_t N>
using tensor = xt::pytensor<T, N>;

#else

/**
 * Fixed (static) rank array.
 */
template <typename T, size_t N>
using tensor = xt::xtensor<T, N>;

#endif

} // namespace array_type

} // namespace GooseEPM

#endif
