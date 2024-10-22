cmake_minimum_required(VERSION 3.10)
project(rk_solver_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 REQUIRED)

include_directories(include)

add_library(rk_solver STATIC
    src/rk_solver.cpp
    src/rk23.cpp
    src/rk45.cpp
    src/rk_step.cpp
    src/dense_output.cpp
    src/solve_ivp.cpp
)

pybind11_add_module(rk_solver_cpp pybind/rk_solver_binding.cpp)
target_link_libraries(rk_solver_cpp PRIVATE rk_solver)