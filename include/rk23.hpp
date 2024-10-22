#ifndef RK23_HPP
#define RK23_HPP

#include "rk_solver.hpp"
#include <array>

namespace rk_solver {

class RK23 : public RungeKutta {
public:
    RK23(OdeFunction fun, double t0, const Vector& y0, double t_bound,
         double max_step = std::numeric_limits<double>::infinity(),
         double rtol = 1e-3, double atol = 1e-6, bool vectorized = false,
         double first_step = 0.0);

protected:
    bool _step_impl() override;
    void _dense_output_impl() override;
    double _estimate_error(const Vector& y_new, const Vector& f_new, const Matrix& K) override;

private:
    static constexpr int order = 2;
    static constexpr int n_stages = 3;
    static constexpr std::array<double, 2> C = {1.0 / 2.0, 3.0 / 4.0};
    static constexpr std::array<std::array<double, 2>, 2> A = {{
        {1.0 / 2.0},
        {0.0, 3.0 / 4.0}
    }};
    static constexpr std::array<double, 3> B = {2.0 / 9.0, 1.0 / 3.0, 4.0 / 9.0};
    static constexpr std::array<double, 4> E = {5.0 / 72.0, -1.0 / 12.0, -1.0 / 9.0, 1.0 / 8.0};
    static constexpr std::array<std::array<double, 3>, 4> P = {{
        {1.0, -4.0 / 3.0, 5.0 / 9.0},
        {0.0, 1.0, -2.0 / 3.0},
        {0.0, 4.0 / 3.0, -8.0 / 9.0},
        {0.0, -1.0, 1.0}
    }};
};

} // namespace rk_solver

#endif // RK23_HPP