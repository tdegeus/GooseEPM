# GooseEPM cmake module
#
# This module sets the target:
#
#   GooseEPM
#
# In addition, it sets the following variables:
#
#   GooseEPM_FOUND - true if GooseEPM found
#   GooseEPM_VERSION - GooseEPM's version
#   GooseEPM_INCLUDE_DIRS - the directory containing GooseEPM headers
#
# The following support targets are defined to simplify things:
#
#   GooseEPM::compiler_warnings - enable compiler warnings
#   GooseEPM::assert - enable GooseEPM assertions
#   GooseEPM::debug - enable all assertions (slow)

include(CMakeFindDependencyMacro)

# Define target "GooseEPM"

if(NOT TARGET GooseEPM)
    include("${CMAKE_CURRENT_LIST_DIR}/GooseEPMTargets.cmake")
    get_target_property(GooseEPM_INCLUDE_DIRS GooseEPM INTERFACE_INCLUDE_DIRECTORIES)
endif()

# Find dependencies

find_dependency(xtensor)

# Define support target "GooseEPM::compiler_warnings"

if(NOT TARGET GooseEPM::compiler_warnings)
    add_library(GooseEPM::compiler_warnings INTERFACE IMPORTED)
    if(MSVC)
        set_property(
            TARGET GooseEPM::compiler_warnings
            PROPERTY INTERFACE_COMPILE_OPTIONS
            /W4)
    else()
        set_property(
            TARGET GooseEPM::compiler_warnings
            PROPERTY INTERFACE_COMPILE_OPTIONS
            -Wall -Wextra -pedantic -Wno-unknown-pragmas)
    endif()
endif()

# Define support target "GooseEPM::warnings"

if(NOT TARGET GooseEPM::warnings)
    add_library(GooseEPM::warnings INTERFACE IMPORTED)
    set_property(
        TARGET GooseEPM::warnings
        PROPERTY INTERFACE_COMPILE_DEFINITIONS
        GOOSEEPM_ENABLE_WARNING_PYTHON)
endif()

# Define support target "GooseEPM::assert"

if(NOT TARGET GooseEPM::assert)
    add_library(GooseEPM::assert INTERFACE IMPORTED)
    set_property(
        TARGET GooseEPM::assert
        PROPERTY INTERFACE_COMPILE_DEFINITIONS
        GOOSEEPM_ENABLE_ASSERT)
endif()

# Define support target "GooseEPM::debug"

if(NOT TARGET GooseEPM::debug)
    add_library(GooseEPM::debug INTERFACE IMPORTED)
    set_property(
        TARGET GooseEPM::debug
        PROPERTY INTERFACE_COMPILE_DEFINITIONS
        XTENSOR_ENABLE_ASSERT PRRNG_ENABLE_ASSERT GOOSEEPM_ENABLE_ASSERT)
endif()
