#ifndef RK_SOLVER_HPP
#define RK_SOLVER_HPP

#include <vector>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include "dense_output.hpp"

namespace rk_solver {

// Forward declarations
class RungeKutta;
class DenseOutput;

// Define basic types if not already defined in dense_output.hpp
#ifndef RK_SOLVER_TYPES_DEFINED
#define RK_SOLVER_TYPES_DEFINED
using Vector = std::vector<double>;
using Matrix = std::vector<Vector>;
using OdeFunction = std::function<Vector(double, const Vector&)>;
#endif

using EventFunction = std::function<double(double t, const Vector& y)>;

struct Event {
    EventFunction function;
    bool terminal = false;
    double direction = 0.0;
};

class OdeSolution {
public:
    // Modified constructor to take ownership of interpolants
    OdeSolution(std::vector<double> ts, 
                std::vector<std::unique_ptr<DenseOutput>>&& interpolants)
        : ts_(std::move(ts)), interpolants_(std::move(interpolants)) {}

    // Delete copy constructor and assignment
    OdeSolution(const OdeSolution&) = delete;
    OdeSolution& operator=(const OdeSolution&) = delete;

    // Allow moving
    OdeSolution(OdeSolution&&) = default;
    OdeSolution& operator=(OdeSolution&&) = default;

    Vector operator()(double t) const;

private:
    std::vector<double> ts_;
    std::vector<std::unique_ptr<DenseOutput>> interpolants_;
};

/**
 * @brief Base class for ODE solvers.
 */
class OdeSolver {
public:
    OdeSolver(OdeFunction fun, double t0, const Vector& y0, double t_bound, 
              bool vectorized = false, bool support_complex = false);

    virtual ~OdeSolver() = default;

    virtual bool step() = 0;
    virtual void integrate(double t) = 0;
    virtual void compute_dense_output() = 0;  // Renamed from dense_output()

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
    void compute_dense_output() override;
    std::unique_ptr<DenseOutput> get_dense_output() const { 
        return dense_output_sol ? std::make_unique<RkDenseOutput>(
            dynamic_cast<RkDenseOutput&>(*dense_output_sol)) : nullptr;
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
    std::unique_ptr<DenseOutput> dense_output_sol;

    static constexpr double MIN_FACTOR = 0.2;
    static constexpr double MAX_FACTOR = 10;
    static constexpr double SAFETY = 0.9;

    void _adjust_step(double error_norm, double safety = SAFETY);
};

// struct SolveIvpResult {
//     std::vector<double> t;
//     std::vector<std::vector<double>> y;
//     std::unique_ptr<OdeSolution> sol;  // Using unique_ptr for OdeSolution
//     std::vector<std::vector<double>> t_events;
//     size_t nfev = 0;
//     size_t njev = 0;
//     size_t nlu = 0;
//     int status = 0;
//     std::string message;
//     bool success = false;
// };

// // Modified solve_ivp implementation to handle move semantics
// SolveIvpResult solve_ivp(
//     RungeKutta* solver,
//     std::vector<double> t_eval,
//     std::vector<Event> events = {},
//     bool dense_output = false
// );

} // namespace rk_solver

#endif // RK_SOLVER_HPP