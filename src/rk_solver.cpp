#include "rk_solver.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <limits>
#include <cmath>
#include <algorithm>
#include <limits>

namespace rk_solver {

double norm(const Vector& v) {
    double sum = 0.0;
    for (const auto& x : v) {
        sum += x * x;
    }
    return std::sqrt(sum);
}

Vector scale_vector(const Vector& y, const Vector& scale) {
    Vector result(y.size());
    for (size_t i = 0; i < y.size(); ++i) {
        result[i] = y[i] / scale[i];
    }
    return result;
}

Vector add_vectors(const Vector& a, const Vector& b) {
    Vector result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] + b[i];
    }
    return result;
}

Vector subtract_vectors(const Vector& a, const Vector& b) {
    Vector result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] - b[i];
    }
    return result;
}

Vector multiply_vector(const Vector& v, double scalar) {
    Vector result(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        result[i] = v[i] * scalar;
    }
    return result;
}

double select_initial_step(const OdeFunction& fun, double t0, const Vector& y0, 
                           const Vector& f0, int direction, int order, 
                           double rtol, double atol) {
    if (y0.empty()) {
        return std::numeric_limits<double>::infinity();
    }

    Vector scale(y0.size());
    for (size_t i = 0; i < y0.size(); ++i) {
        scale[i] = atol + std::abs(y0[i]) * rtol;
    }

    double d0 = norm(scale_vector(y0, scale));
    double d1 = norm(scale_vector(f0, scale));

    double h0;
    if (d0 < 1e-5 || d1 < 1e-5) {
        h0 = 1e-6;
    } else {
        h0 = 0.01 * d0 / d1;
    }

    Vector y1 = add_vectors(y0, multiply_vector(f0, h0 * direction));
    Vector f1 = fun(t0 + h0 * direction, y1);

    double d2 = norm(scale_vector(subtract_vectors(f1, f0), scale)) / h0;

    double h1;
    if (d1 <= 1e-15 && d2 <= 1e-15) {
        h1 = std::max(1e-6, h0 * 1e-3);
    } else {
        h1 = std::pow(0.01 / std::max(d1, d2), 1.0 / (order + 1));
    }

    return std::min(100 * h0, h1);
}

OdeSolver::OdeSolver(OdeFunction fun, double t0, const Vector& y0, double t_bound, 
                     bool vectorized, bool support_complex)
    : fun(std::move(fun)), t(t0), y(y0), t_bound(t_bound), 
      vectorized(vectorized), support_complex(support_complex),
      t_old(t0), y_old(y0), nfev(0), njev(0), nlu(0), status("running") {
    direction = (t_bound > t0) ? 1 : -1;
    n = y0.size();
}

RungeKutta::RungeKutta(OdeFunction fun, double t0, const Vector& y0, double t_bound,
                       double max_step, double rtol, double atol, bool vectorized,
                       double first_step, int order)
    : OdeSolver(std::move(fun), t0, y0, t_bound, vectorized),
      order(order), max_step(max_step), rtol(rtol), atol(atol) {
    f = this->fun(t, y);
    nfev++;

    if (first_step == 0.0) {
        h_abs = select_initial_step(this->fun, t, y, f, direction, order, rtol, atol);
    } else {
        h_abs = std::abs(first_step);
    }

    status = "running";
}

bool RungeKutta::step() {
    t_old = t;
    y_old = y;
    
    bool success = false;
    try {
        success = _step_impl();
        
        if (success) {
            _dense_output_impl();
        } else {
            t = t_old;
            y = y_old;
        }
    } catch (const std::exception& e) {
        throw std::runtime_error("Step failed: " + std::string(e.what()));
    }
    
    return success;
}

// void RungeKutta::integrate(double t_target) {
//     if (status != "running") {
//         throw std::runtime_error("Solver is not running. Current status: " + status);
//     }

//     if (std::abs(t_target - t) < 10 * std::numeric_limits<double>::epsilon()) {
//         return;
//     }

//     try {
//         while ((t_target - t) * direction > 0) {
//             if (!step()) {
//                 status = "failed";
//                 throw std::runtime_error("Integration step failed");
//             }
            
//             if (std::abs(t - t_target) < 10 * std::numeric_limits<double>::epsilon()) {
//                 break;
//             }
//         }

//         if ((t_bound - t) * direction <= 0) {
//             status = "finished";
//         }
//     } catch (const std::exception& e) {
//         status = "failed";
//         throw std::runtime_error("Integration failed: " + std::string(e.what()));
//     }
// }

void RungeKutta::integrate(double t_target) {
    if (status != "running") {
        throw std::runtime_error("Solver is not running. Current status: " + status);
    }

    while ((t_target - t) * direction > 0) {
        if (direction * (t + h_abs - t_target) > 0) {
            h_abs = std::abs(t_target - t);
        }
        if (!step()) {
            status = "failed";
            throw std::runtime_error("Integration step failed");
        }
        if (std::abs(t - t_target) < 10 * std::numeric_limits<double>::epsilon()) {
            break;
        }
    }

    if ((t_bound - t) * direction <= 0) {
        status = "finished";
    }
}

// double RungeKutta::_estimate_error(const Vector& K, const Vector& E) {
//     Vector scale(n);
//     for (size_t i = 0; i < n; ++i) {
//         scale[i] = atol + std::max(std::abs(y[i]), std::abs(y[i] + K[i])) * rtol;
//     }
//     Vector error(n);
//     for (size_t i = 0; i < n; ++i) {
//         error[i] = std::abs(K[i] * E[i]) / scale[i];
//     }
//     return norm(error);
// }

// double RungeKutta::_estimate_error(const Vector& y_new, const Vector& f_new, const Vector& K) {
//     double error_norm = 0.0;
//     for (size_t i = 0; i < n; ++i) {
//         double sc = atol + std::max(std::abs(y[i]), std::abs(y_new[i])) * rtol;
//         double error = std::abs(h_abs * (K[0][i] * E[0] + K[1][i] * E[1] + K[2][i] * E[2] + f_new[i] * E[3])) / sc;
//         error_norm += error * error;
//     }
//     return std::sqrt(error_norm / n);
// }

void RungeKutta::_adjust_step(double error_norm, double safety) {
    if (error_norm == 0.0) {
        h_abs *= MAX_FACTOR;
    } else if (error_norm < 1) {
        double factor = std::min(MAX_FACTOR, std::max(1.0, safety * std::pow(error_norm, -1.0 / (order + 1))));
        h_abs *= factor;
    } else {
        double factor = std::max(MIN_FACTOR, safety * std::pow(error_norm, -1.0 / (order + 1)));
        h_abs *= factor;
    }
}

} // namespace rk_solver