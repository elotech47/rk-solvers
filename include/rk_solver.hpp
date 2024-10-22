#ifndef RK_SOLVER_HPP
#define RK_SOLVER_HPP

#include "rk_step.hpp"
#include "dense_output.hpp"
#include <limits>
#include <string>

namespace rk_solver {

/**
 * @brief Base class for ODE solvers.
 * 
 * This class defines the interface for all ODE solvers in the library.
 */

class OdeSolver {
public:
    OdeSolver(OdeFunction fun, double t0, const Vector& y0, double t_bound, 
              bool vectorized = false, bool support_complex = false);

    virtual ~OdeSolver() = default;

    virtual bool step() = 0;
    virtual void integrate(double t) = 0;

    double get_t() const { return t; }
    const Vector& get_y() const { return y; }

protected:
    OdeFunction fun;
    double t;
    Vector y;
    double t_bound;
    int direction;
    size_t n;
    bool vectorized;
    bool support_complex;
    double t_old;
    Vector y_old;
    size_t nfev;
    size_t njev;
    size_t nlu;
    std::string status;

    static constexpr double TOO_SMALL_STEP = -1;
};

/**
 * @brief Base class for Runge-Kutta methods.
 * 
 * This class implements common functionality for Runge-Kutta methods.
 * It provides adaptive step size control and dense output capabilities.
 * 
 * @param fun The right-hand side of the ODE system. It should be a callable that takes
 *            (t, y) as arguments and returns dy/dt.
 * @param t0 The initial time.
 * @param y0 The initial state (should be a std::vector<double>).
 * @param t_bound The boundary time - the integration won't continue beyond it.
 * @param max_step Maximum allowed step size (default is infinity).
 * @param rtol Relative tolerance for error control (default is 1e-3).
 * @param atol Absolute tolerance for error control (default is 1e-6).
 * @param vectorized Whether the fun is implemented in a vectorized fashion (default is false).
 * @param first_step Initial step size (default is 0.0, which means the algorithm will guess).
 */

class RungeKutta : public OdeSolver {
public:
    RungeKutta(OdeFunction fun, double t0, const Vector& y0, double t_bound,
               double max_step = std::numeric_limits<double>::infinity(),
               double rtol = 1e-3, double atol = 1e-6, bool vectorized = false,
               double first_step = 0.0, int order = 0);

    int get_order() const { return order; }

    bool step() override;
    void integrate(double t) override;
    std::unique_ptr<DenseOutput> get_dense_output() const { 
        return dense_output ? std::make_unique<RkDenseOutput>(
            dynamic_cast<RkDenseOutput&>(*dense_output)) : nullptr;
    }

protected:
    
    virtual bool _step_impl() = 0;
    virtual void _dense_output_impl() = 0;
    virtual double _estimate_error(const Vector& y_new, const Vector& f_new, const Matrix& K) = 0;

    int order;
    double max_step;
    double rtol;
    double atol;
    Vector f;
    double h_abs;
    Matrix K;
    std::unique_ptr<DenseOutput> dense_output;

    static constexpr double MIN_FACTOR = 0.2;
    static constexpr double MAX_FACTOR = 10;
    static constexpr double SAFETY = 0.9;

    // double _estimate_error(const Vector& K, const Vector& E);
    // virtual double _estimate_error(const Vector& y_new, const Vector& f_new, const Matrix& K) = 0;
    void _adjust_step(double error_norm, double safety = SAFETY);
};

} // namespace rk_solver

#endif // RK_SOLVER_HPP