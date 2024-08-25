# Require CMake 3.15+ (matching scikit-build-core) Use new versions of all
# policies up to CMake 3.27
cmake_minimum_required(VERSION 3.15...3.27)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Scikit-build-core sets these values for you, or you can just hard-code the
# name and version.
project(
    ${SKBUILD_PROJECT_NAME}
    VERSION ${SKBUILD_PROJECT_VERSION}
    LANGUAGES CXX)

# Find the module development requirements (requires FindPython from 3.17 or
# scikit-build-core's built-in backport)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(OpenCV REQUIRED)
add_subdirectory(third-party/pybind11)

include_directories(include)
file(GLOB SOURCES "src/*.cpp")

python_add_library(_core MODULE ${SOURCES} WITH_SOABI)
target_link_libraries(_core PRIVATE pybind11::headers ${OpenCV_LIBS})
target_include_directories(_core PRIVATE ${OpenCV_INCLUDE_DIRS})

# This is passing in the version as a define just as an example
target_compile_definitions(_core PRIVATE VERSION_INFO=${PROJECT_VERSION})

# The install directory is the output (wheel) directory
install(TARGETS _core DESTINATION mcerl)
