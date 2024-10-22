#ifndef RK45_HPP
#define RK45_HPP

#include "rk_solver.hpp"
#include <array>

namespace rk_solver {

class RK45 : public RungeKutta {
public:
    RK45(OdeFunction fun, double t0, const Vector& y0, double t_bound,
         double max_step = std::numeric_limits<double>::infinity(),
         double rtol = 1e-3, double atol = 1e-6, bool vectorized = false,
         double first_step = 0.0);

protected:
    bool _step_impl() override;
    void _dense_output_impl() override;
    double _estimate_error(const Vector& y_new, const Vector& f_new, const Matrix& K) override;

private:
    static constexpr int order = 4;
    static constexpr int n_stages = 6;
    static constexpr std::array<double, 5> C = {1.0/5, 3.0/10, 4.0/5, 8.0/9, 1.0};
    static constexpr std::array<std::array<double, 5>, 5> A = {{
        {1.0/5},
        {3.0/40, 9.0/40},
        {44.0/45, -56.0/15, 32.0/9},
        {19372.0/6561, -25360.0/2187, 64448.0/6561, -212.0/729},
        {9017.0/3168, -355.0/33, 46732.0/5247, 49.0/176, -5103.0/18656}
    }};
    static constexpr std::array<double, 6> B = {35.0/384, 0, 500.0/1113, 125.0/192, -2187.0/6784, 11.0/84};
    static constexpr std::array<double, 7> E = {
        71.0/57600, 0, -71.0/16695, 71.0/1920, -17253.0/339200, 22.0/525, -1.0/40
    };
    static constexpr std::array<std::array<double, 4>, 7> P = {{
        {1, -8048581381.0/2820520608, 8663915743.0/2820520608, -12715105075.0/11282082432},
        {0, 0, 0, 0},
        {0, 131558114200.0/32700410799, -68118460800.0/10900136933, 87487479700.0/32700410799},
        {0, -1754552775.0/470086768, 14199869525.0/1410260304, -10690763975.0/1880347072},
        {0, 127303824393.0/49829197408, -318862633887.0/49829197408, 701980252875.0/199316789632},
        {0, -282668133.0/205662961, 2019193451.0/616988883, -1453857185.0/822651844},
        {0, 40617522.0/29380423, -110615467.0/29380423, 69997945.0/29380423}
    }};

};

} // namespace rk_solver

#endif // RK45_HPP