cmake_minimum_required(VERSION 3.15)
project(${SKBUILD_PROJECT_NAME} LANGUAGES C CXX)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_core MODULE src/wrapper.cpp)
install(TARGETS _core DESTINATION ${SKBUILD_PROJECT_NAME})
