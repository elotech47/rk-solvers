#ifndef SOLVE_IVP_HPP
#define SOLVE_IVP_HPP

#include "rk_solver.hpp"
#include <vector>

namespace rk_solver {

struct SolveIvpResult {
    std::vector<double> t;
    std::vector<std::vector<double>> y;
    std::string message;
    bool success;
};

SolveIvpResult solve_ivp(RungeKutta* solver, const std::vector<double>& t_eval);

} // namespace rk_solver

#endif // SOLVE_IVP_HPP