// solve_ivp.hpp
#ifndef SOLVE_IVP_HPP
#define SOLVE_IVP_HPP

#include "rk_solver.hpp"
#include <vector>
#include <memory>
#include <string>

namespace rk_solver {

// Forward declaration of OdeSolution if not already defined
class OdeSolution;

struct SolveIvpResult {
    std::vector<double> t;
    std::vector<std::vector<double>> y;
    std::unique_ptr<OdeSolution> sol;  // Using unique_ptr for OdeSolution
    std::vector<std::vector<double>> t_events;
    size_t nfev = 0;
    size_t njev = 0;
    size_t nlu = 0;
    int status = 0;
    std::string message;
    bool success = false;
};

SolveIvpResult solve_ivp(
    RungeKutta* solver,
    std::vector<double> t_eval,
    std::vector<Event> events = {},
    bool dense_output = false
);

} // namespace rk_solver

#endif // SOLVE_IVP_HPP
