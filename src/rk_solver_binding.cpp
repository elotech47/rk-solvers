#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "rk_solver.hpp"
#include "rk23.hpp"
#include "rk45.hpp"
#include "solve_ivp.hpp"
namespace py = pybind11;

// Define a wrapper for the ODE function that converts Python objects to C++ types
rk_solver::Vector ode_function_wrapper(const py::object& fun, double t, const rk_solver::Vector& y) {
    py::array_t<double> y_array(y.size(), y.data());
    py::object result = fun(t, y_array);
    return result.cast<rk_solver::Vector>();
}

PYBIND11_MODULE(rk_solver_cpp, m) {
    m.doc() = "C++ implementation of Runge-Kutta solvers";

    py::class_<rk_solver::OdeSolver>(m, "OdeSolver")
        .def("step", &rk_solver::OdeSolver::step)
        .def("integrate", &rk_solver::OdeSolver::integrate)
        .def("get_t", &rk_solver::OdeSolver::get_t)
        .def("get_y", &rk_solver::OdeSolver::get_y);

    py::class_<rk_solver::RungeKutta, rk_solver::OdeSolver>(m, "RungeKutta");

    py::class_<rk_solver::RK23, rk_solver::RungeKutta>(m, "RK23")
        .def(py::init([](const py::object& fun, double t0, const py::array_t<double>& y0, double t_bound,
                         double max_step, double rtol, double atol, bool vectorized, double first_step) {
            auto wrapped_fun = [fun](double t, const rk_solver::Vector& y) {
                return ode_function_wrapper(fun, t, y);
            };
            rk_solver::Vector y0_vec(y0.data(), y0.data() + y0.size());
            return new rk_solver::RK23(wrapped_fun, t0, y0_vec, t_bound, max_step, rtol, atol, vectorized, first_step);
        }),
        py::arg("fun"), py::arg("t0").noconvert(), py::arg("y0").noconvert(), py::arg("t_bound").noconvert(),
        py::arg("max_step") = std::numeric_limits<double>::infinity(),
        py::arg("rtol") = 1e-3, py::arg("atol") = 1e-6,
        py::arg("vectorized") = false, py::arg("first_step") = 0.0);

    py::class_<rk_solver::RK45, rk_solver::RungeKutta>(m, "RK45")
        .def(py::init([](const py::object& fun, double t0, const py::array_t<double>& y0, double t_bound,
                         double max_step, double rtol, double atol, bool vectorized, double first_step) {
            auto wrapped_fun = [fun](double t, const rk_solver::Vector& y) {
                return ode_function_wrapper(fun, t, y);
            };
            rk_solver::Vector y0_vec(y0.data(), y0.data() + y0.size());
            return new rk_solver::RK45(wrapped_fun, t0, y0_vec, t_bound, max_step, rtol, atol, vectorized, first_step);
        }),
        py::arg("fun"), py::arg("t0").noconvert(), py::arg("y0").noconvert(), py::arg("t_bound").noconvert(),
        py::arg("max_step") = std::numeric_limits<double>::infinity(),
        py::arg("rtol") = 1e-3, py::arg("atol") = 1e-6,
        py::arg("vectorized") = false, py::arg("first_step") = 0.0);

    py::class_<rk_solver::DenseOutput>(m, "DenseOutput")
        .def("__call__", &rk_solver::DenseOutput::call);

    m.def("solve_ivp", [](rk_solver::RungeKutta* solver, const py::array_t<double>& t_eval) {
        std::vector<double> t_eval_vec(t_eval.data(), t_eval.data() + t_eval.size());
        auto result = rk_solver::solve_ivp(solver, t_eval_vec);
        
        py::dict py_result;
        py_result["t"] = result.t;
        py_result["y"] = result.y;
        py_result["message"] = result.message;
        py_result["success"] = result.success;
        
        return py_result;
    });
}