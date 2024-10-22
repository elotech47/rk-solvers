#ifndef DENSE_OUTPUT_HPP
#define DENSE_OUTPUT_HPP

#include <vector>

namespace rk_solver {

using Vector = std::vector<double>;
using Matrix = std::vector<Vector>;

class DenseOutput {
public:
    DenseOutput(double t_old, double t);
    virtual ~DenseOutput() = default;

    virtual Vector call(double t) = 0;

protected:
    double t_old;
    double t;
};

class RkDenseOutput : public DenseOutput {
public:
    RkDenseOutput(double t_old, double t, const Vector& y_old, const Matrix& Q);
    Vector call(double t) override;

private:
    Vector y_old;
    Matrix Q;
    size_t order;
};

} // namespace rk_solver

#endif // DENSE_OUTPUT_HPP