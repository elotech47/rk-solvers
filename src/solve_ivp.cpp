#include "solve_ivp.hpp"
#include <stdexcept>
#include <iostream>

namespace rk_solver {

SolveIvpResult solve_ivp(RungeKutta* solver, const std::vector<double>& t_eval) {
    SolveIvpResult result;
    result.t = t_eval;
    result.y.resize(t_eval.size());
    result.success = true;
    result.message = "Successful integration.";

    try {
        for (size_t i = 0; i < t_eval.size(); ++i) {
            solver->integrate(t_eval[i]);
            result.y[i] = solver->get_y();
        }
    } catch (const std::exception& e) {
        result.success = false;
        result.message = std::string("Integration failed: ") + e.what();
    }

    return result;
}

} // namespace rk_solver