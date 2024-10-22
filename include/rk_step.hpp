#ifndef RK_STEP_HPP
#define RK_STEP_HPP

#include <vector>
#include <array>
#include <functional>

namespace rk_solver {

using Vector = std::vector<double>;
using Matrix = std::vector<Vector>;
using OdeFunction = std::function<Vector(double, const Vector&)>;

struct RkStepResult {
    Vector y_new;
    Vector f_new;
};

// Generic template declaration
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
);

// Declare specializations
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
);

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
);

} // namespace rk_solver

#endif // RK_STEP_HPP