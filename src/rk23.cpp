#include "rk23.hpp"
#include "rk_step.hpp"
#include "dense_output.hpp"
#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace rk_solver {

RK23::RK23(OdeFunction fun, double t0, const Vector& y0, double t_bound,
           double max_step, double rtol, double atol, bool vectorized,
           double first_step)
    : RungeKutta(std::move(fun), t0, y0, t_bound, max_step, rtol, atol, vectorized, first_step, order) {
    K.resize(n_stages + 1, Vector(n));
}

double RK23::_estimate_error(const Vector& y_new, const Vector& f_new, const Matrix& K) {
    double error_norm_sq = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double sc = atol + std::max(std::abs(y[i]), std::abs(y_new[i])) * rtol;
        double error = std::abs(h_abs * (E[0] * K[0][i] + E[1] * K[1][i] + E[2] * K[2][i] + E[3] * f_new[i])) / sc;
        error_norm_sq += error * error;
    }
    return std::sqrt(error_norm_sq / n);
}

bool RK23::_step_impl() {
    double t_new;
    Vector y_new, f_new;
    bool step_accepted = false;

    while (!step_accepted) {
        if (h_abs < 10 * std::numeric_limits<double>::epsilon() * std::abs(t)) {
            return false;
        }

        h_abs = std::min(h_abs, max_step);
        double h = h_abs * direction;

        t_new = t + h;
        if (direction * (t_new - t_bound) > 0) {
            t_new = t_bound;
            h = t_new - t;
            h_abs = std::abs(h);
        }

        try {
            auto result = rk_step<2, 2>(fun, t, y, f, h, A, B, C, E, K, rtol, atol);
            y_new = result.y_new;
            f_new = result.f_new;
            double error_norm = _estimate_error(y_new, f_new, K);

            nfev += n_stages;

            if (error_norm <= 1.0) {
                step_accepted = true;
                t = t_new;
                y = y_new;
                f = f_new;
            }

            // Step size adjustment
            double factor = std::min(MAX_FACTOR, std::max(MIN_FACTOR, SAFETY * std::pow(error_norm, -1.0 / (order + 1))));
            if (step_accepted) {
                h_abs *= factor;
            } else {
                h_abs *= std::max(0.1, factor);  // More conservative reduction for rejected steps
            }

        } catch (const std::exception& e) {
            throw std::runtime_error("Exception in rk_step: " + std::string(e.what()));
        }
    }

    return true;
}

void RK23::_dense_output_impl() {
    try {
        Matrix Q(n, Vector(4, 0.0));
        
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < 4; ++j) {
                for (size_t k = 0; k < std::min(size_t(4), K.size()); ++k) {
                    if (j < P[k].size() && i < K[k].size()) {
                        Q[i][j] += P[k][j] * K[k][i];
                    }
                }
            }
        }
        
        dense_output = std::make_unique<RkDenseOutput>(t_old, t, y_old, Q);
    } catch (const std::exception& e) {
        throw std::runtime_error("Exception in _dense_output_impl: " + std::string(e.what()));
    }
}

} // namespace rk_solver