#include "rk_step.hpp"
#include <cstring>

namespace rk_solver {

// Generic implementation
template<size_t N, size_t M>
RkStepResult rk_step(
    const OdeFunction& fun,
    double t,
    const Vector& y,
    const Vector& f,
    double h,
    const std::array<std::array<double, N>, M>& A,
    const std::array<double, M+1>& B,
    const std::array<double, M>& C,
    const std::array<double, M+2>& E,
    Matrix& K,
    double rtol,
    double atol
) {
    size_t n_stages = M + 1;
    size_t n = y.size();

    if (K.size() != n_stages || K[0].size() != n) {
        K.resize(n_stages, Vector(n));
    }

    K[0] = f;
    Vector y_stage(n);

    for (size_t s = 1; s < n_stages; ++s) {
        for (size_t i = 0; i < n; ++i) {
            y_stage[i] = y[i];
            for (size_t j = 0; j < s; ++j) {
                y_stage[i] += h * A[s-1][j] * K[j][i];
            }
        }
        K[s] = fun(t + C[s-1] * h, y_stage);
    }

    Vector y_new = y;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n_stages; ++j) {
            y_new[i] += h * B[j] * K[j][i];
        }
    }
    Vector f_new = fun(t + h, y_new);

    return {std::move(y_new), std::move(f_new)};
}

// Specialization for RK23
template<>
RkStepResult rk_step<2, 2>(
    const OdeFunction& fun,
    double t,
    const Vector& y,
    const Vector& f,
    double h,
    const std::array<std::array<double, 2>, 2>& A,
    const std::array<double, 3>& B,
    const std::array<double, 2>& C,
    const std::array<double, 4>& E,
    Matrix& K,
    double rtol,
    double atol
) {
    constexpr size_t n_stages = 3;
    const size_t n = y.size();

    if (K.size() != n_stages || K[0].size() != n) {
        K.resize(n_stages, Vector(n));
    }

    std::memcpy(K[0].data(), f.data(), n * sizeof(double));

    Vector y_stage(n);
    for (size_t s = 1; s < n_stages; ++s) {
        std::memcpy(y_stage.data(), y.data(), n * sizeof(double));
        
        for (size_t i = 0; i < n; ++i) {
            double stage_sum = 0.0;
            for (size_t j = 0; j < s; ++j) {
                stage_sum += A[s-1][j] * K[j][i];
            }
            y_stage[i] += h * stage_sum;
        }
        
        K[s] = fun(t + C[s-1] * h, y_stage);
    }

    Vector y_new(n);
    std::memcpy(y_new.data(), y.data(), n * sizeof(double));
    for (size_t i = 0; i < n; ++i) {
        double stage_sum = 0.0;
        for (size_t j = 0; j < n_stages; ++j) {
            stage_sum += B[j] * K[j][i];
        }
        y_new[i] += h * stage_sum;
    }

    Vector f_new = fun(t + h, y_new);

    return {std::move(y_new), std::move(f_new)};
}

// Specialization for RK45 (optimized version)
template<>
RkStepResult rk_step<5, 5>(
    const OdeFunction& fun,
    double t,
    const Vector& y,
    const Vector& f,
    double h,
    const std::array<std::array<double, 5>, 5>& A,
    const std::array<double, 6>& B,
    const std::array<double, 5>& C,
    const std::array<double, 7>& E,
    Matrix& K,
    double rtol,
    double atol
) {
    constexpr size_t n_stages = 6;
    const size_t n = y.size();

    if (K.size() != n_stages || K[0].size() != n) {
        K.resize(n_stages, Vector(n));
    }

    std::array<double, 5> stage_sum;
    Vector y_new(n);
    Vector y_stage(n);

    std::memcpy(K[0].data(), f.data(), n * sizeof(double));

    for (size_t s = 1; s < n_stages; ++s) {
        std::memcpy(y_stage.data(), y.data(), n * sizeof(double));
        
        for (size_t i = 0; i < n; ++i) {
            stage_sum[0] = 0.0;
            for (size_t j = 0; j < s; ++j) {
                stage_sum[0] += A[s-1][j] * K[j][i];
            }
            y_stage[i] += h * stage_sum[0];
        }
        
        K[s] = fun(t + C[s-1] * h, y_stage);
    }

    std::memcpy(y_new.data(), y.data(), n * sizeof(double));
    for (size_t i = 0; i < n; ++i) {
        stage_sum[0] = 0.0;
        for (size_t j = 0; j < n_stages; ++j) {
            stage_sum[0] += B[j] * K[j][i];
        }
        y_new[i] += h * stage_sum[0];
    }

    Vector f_new = fun(t + h, y_new);

    return {std::move(y_new), std::move(f_new)};
}

} // namespace rk_solver