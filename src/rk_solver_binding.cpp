#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "rk_solver.hpp"
#include "rk23.hpp"
#include "rk45.hpp"
#include "solve_ivp.hpp"

namespace py = pybind11;

// Wrapper for ODE function
rk_solver::Vector ode_function_wrapper(const py::object& fun, double t, const rk_solver::Vector& y) {
    py::array_t<double> y_array(y.size(), y.data());
    py::object result = fun(t, y_array);
    return result.cast<rk_solver::Vector>();
}

// Wrapper for event function
double event_function_wrapper(const py::object& event_fun, double t, const rk_solver::Vector& y) {
    py::array_t<double> y_array(y.size(), y.data());
    py::object result = event_fun(t, y_array);
    return result.cast<double>();
}

PYBIND11_MODULE(rk_solver_cpp, m) {
    m.doc() = "C++ implementation of Runge-Kutta solvers";

    // Bind the Event class
    py::class_<rk_solver::Event>(m, "Event")
        .def(py::init<>())
        .def_readwrite("terminal", &rk_solver::Event::terminal)
        .def_readwrite("direction", &rk_solver::Event::direction)
        .def("__call__", [](const rk_solver::Event& e, double t, const py::array_t<double>& y) {
            rk_solver::Vector y_vec(y.data(), y.data() + y.size());
            return e.function(t, y_vec);
        });

    // Bind the OdeSolution class
    py::class_<rk_solver::OdeSolution>(m, "OdeSolution")
        .def("__call__", [](const rk_solver::OdeSolution& sol, double t) {
            return sol(t);
        });

    // Original OdeSolver bindings
    py::class_<rk_solver::OdeSolver>(m, "OdeSolver")
        .def("step", &rk_solver::OdeSolver::step)
        .def("integrate", &rk_solver::OdeSolver::integrate)
        .def("compute_dense_output", &rk_solver::OdeSolver::compute_dense_output)
        .def("get_t", &rk_solver::OdeSolver::get_t)
        .def("get_y", &rk_solver::OdeSolver::get_y);

    py::class_<rk_solver::RungeKutta, rk_solver::OdeSolver>(m, "RungeKutta");

    // RK23 bindings (unchanged)
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

    // RK45 bindings (unchanged)
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

    // Updated solve_ivp binding with full features
    m.def("solve_ivp", [](rk_solver::RungeKutta* solver, 
                         const py::array_t<double>& t_eval,
                         const py::object& events = py::none(),
                         bool dense_output = false) {
        std::vector<double> t_eval_vec(t_eval.data(), t_eval.data() + t_eval.size());
        std::vector<rk_solver::Event> cpp_events;

        // Convert Python events to C++ events
        if (!events.is_none()) {
            py::list event_list = events.cast<py::list>();
            for (const py::handle& event_handle : event_list) {
                py::object event_obj = event_handle.cast<py::object>();
                
                rk_solver::Event cpp_event;
                cpp_event.function = [event_obj](double t, const rk_solver::Vector& y) {
                    return event_function_wrapper(event_obj, t, y);
                };

                // Get event attributes if they exist
                if (py::hasattr(event_obj, "terminal")) {
                    cpp_event.terminal = event_obj.attr("terminal").cast<bool>();
                }
                if (py::hasattr(event_obj, "direction")) {
                    cpp_event.direction = event_obj.attr("direction").cast<double>();
                }

                cpp_events.push_back(cpp_event);
            }
        }

        // Call the C++ solve_ivp
        auto result = rk_solver::solve_ivp(solver, t_eval_vec, cpp_events, dense_output);
        
        // Convert result to Python dictionary
        py::dict py_result;
        py_result["t"] = result.t;
        py_result["y"] = result.y;
        py_result["success"] = result.success;
        py_result["status"] = result.status;
        py_result["message"] = result.message;
        py_result["nfev"] = result.nfev;
        py_result["njev"] = result.njev;
        py_result["nlu"] = result.nlu;

        // Add event times if any
        if (!result.t_events.empty()) {
            py::list t_events;
            for (const auto& event_times : result.t_events) {
                t_events.append(event_times);
            }
            py_result["t_events"] = t_events;
        } else {
            py_result["t_events"] = py::none();
        }

        // Add dense output solution if requested
        if (dense_output && result.sol) {
            py_result["sol"] = result.sol.get();
        } else {
            py_result["sol"] = py::none();
        }
        
        return py_result;
    }, py::arg("solver"), 
       py::arg("t_eval").noconvert(), 
       py::arg("events") = py::none(),
       py::arg("dense_output") = false);
}