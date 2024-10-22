#include "dense_output.hpp"
#include <cmath>

namespace rk_solver {

DenseOutput::DenseOutput(double t_old, double t) : t_old(t_old), t(t) {}

RkDenseOutput::RkDenseOutput(double t_old, double t, const Vector& y_old, const Matrix& Q)
    : DenseOutput(t_old, t), y_old(y_old), Q(Q), order(Q[0].size() - 1) {}

Vector RkDenseOutput::call(double t) {
    double x = (t - t_old) / (this->t - t_old);
    Vector p(order + 1);
    p[0] = 1.0;
    for (size_t i = 1; i <= order; ++i) {
        p[i] = p[i-1] * x;
    }

    Vector y = y_old;
    for (size_t i = 0; i < Q.size(); ++i) {
        double dy = 0.0;
        for (size_t j = 0; j <= order; ++j) {
            dy += Q[i][j] * p[j];
        }
        y[i] += dy * (this->t - t_old);
    }

    return y;
}

} // namespace rk_solver