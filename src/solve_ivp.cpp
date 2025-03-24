#include "solve_ivp.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <numeric>      // for std::iota
#include <limits>

namespace rk_solver {

Vector OdeSolution::operator()(double t) const {
    auto it = std::upper_bound(ts_.begin(), ts_.end(), t);
    if (it == ts_.begin() || it == ts_.end()) {
        throw std::runtime_error("Time out of bounds");
    }
    size_t idx = std::distance(ts_.begin(), it) - 1;
    return interpolants_[idx]->call(t);
}

namespace detail {

std::vector<size_t> find_active_events(
    const std::vector<double>& g_old,
    const std::vector<double>& g_new,
    const std::vector<Event>& events
) {
    std::vector<size_t> active_events;
    for (size_t i = 0; i < events.size(); ++i) {
        const bool up = g_old[i] <= 0 && g_new[i] >= 0;
        const bool down = g_old[i] >= 0 && g_new[i] <= 0;
        
        if ((up && events[i].direction >= 0) || 
            (down && events[i].direction <= 0) || 
            (events[i].direction == 0 && (up || down))) {
            active_events.push_back(i);
        }
    }
    return active_events;
}

double solve_event_equation(
    const Event& event,
    const std::unique_ptr<DenseOutput>& sol,
    double t_old,
    double t
) {
    constexpr double eps = 4 * std::numeric_limits<double>::epsilon();
    constexpr int max_iter = 50;
    
    auto event_func = [&](double t_val) {
        return event.function(t_val, sol->call(t_val));
    };
    
    double a = t_old;
    double b = t;
    double fa = event_func(a);
    double fb = event_func(b);
    
    if (std::abs(fa) < eps) return a;
    if (std::abs(fb) < eps) return b;
    if (fa * fb >= 0) {
        throw std::runtime_error("Root finding failed: signs of f(a) and f(b) must differ");
    }
    
    // Modified regula falsi method
    double c = a;
    double fc = fa;
    bool flag = true;
    int side = 0;
    
    for (int i = 0; i < max_iter; ++i) {
        // Compute new approximation
        double c_new;
        if (flag) {
            c_new = (fb * a - fa * b) / (fb - fa); // interpolation
        } else {
            c_new = 0.5 * (a + b);  // bisection
        }
        
        if (std::abs(c_new - c) < eps * std::abs(c_new)) {
            return c_new;
        }
        
        // Evaluate function at new point
        c = c_new;
        fc = event_func(c);
        
        if (std::abs(fc) < eps) {
            return c;
        }
        
        // Update interval
        if (fc * fb < 0) {
            a = b;
            fa = fb;
            b = c;
            fb = fc;
            flag = true;
            if (side == -1) side = -2;
            side = -1;
        } else {
            b = c;
            fb = fc;
            flag = side != -2;
            if (side == 1) side = 2;
            side = 1;
        }
    }
    
    throw std::runtime_error("Event equation solving did not converge");
}

std::tuple<std::vector<size_t>, std::vector<double>, bool>
handle_events(
    const std::unique_ptr<DenseOutput>& sol,
    const std::vector<Event>& events,
    const std::vector<size_t>& active_events,
    double t_old,
    double t
) {
    std::vector<size_t> event_indices = active_events;
    std::vector<double> roots;
    bool terminate = false;
    
    // Find roots for all active events
    for (size_t idx : active_events) {
        try {
            roots.push_back(solve_event_equation(events[idx], sol, t_old, t));
        } catch (const std::runtime_error& e) {
            // Skip events where root finding fails
            continue;
        }
    }
    
    if (!roots.empty()) {
        // Sort events by time of occurrence
        std::vector<size_t> order(roots.size());
        std::iota(order.begin(), order.end(), 0);
        
        const bool forward = t > t_old;
        std::sort(order.begin(), order.end(),
                 [&](size_t i1, size_t i2) {
                     return forward ? roots[i1] < roots[i2] : roots[i1] > roots[i2];
                 });
        
        // Reorder events and roots
        std::vector<size_t> sorted_indices;
        std::vector<double> sorted_roots;
        for (size_t idx : order) {
            sorted_indices.push_back(event_indices[idx]);
            sorted_roots.push_back(roots[idx]);
        }
        
        event_indices = std::move(sorted_indices);
        roots = std::move(sorted_roots);
        
        // Check for terminal events
        for (size_t i = 0; i < event_indices.size(); ++i) {
            if (events[event_indices[i]].terminal) {
                terminate = true;
                // Keep only events up to and including the terminal event
                event_indices.resize(i + 1);
                roots.resize(i + 1);
                break;
            }
        }
    }
    
    return {event_indices, roots, terminate};
}

} // namespace detail

SolveIvpResult solve_ivp(
    RungeKutta* solver,
    std::vector<double> t_eval,
    std::vector<Event> events,
    bool dense_output
) {
    if (!solver) {
        throw std::invalid_argument("Solver cannot be null");
    }
    
    SolveIvpResult result;
    result.t = std::move(t_eval);
    result.y.resize(result.t.size());
    result.t_events.resize(events.size());
    
    std::vector<double> g_old;
    std::vector<std::unique_ptr<DenseOutput>> interpolants;
    
    // Initialize event values at t0
    if (!events.empty()) {
        g_old.reserve(events.size());
        const Vector& y0 = solver->get_y();
        for (const auto& event : events) {
            g_old.push_back(event.function(solver->get_t(), y0));
        }
    }
    
    try {
        for (size_t i = 0; i < result.t.size(); ++i) {
            const double t_target = result.t[i];
            const double t_current = solver->get_t();
            
            // Skip integration if we're already at the target time
            if (std::abs(t_target - t_current) > 10 * std::numeric_limits<double>::epsilon()) {
                solver->integrate(t_target);
            }
            
            result.y[i] = solver->get_y();
            
            if (dense_output) {
                solver->compute_dense_output();
                interpolants.push_back(solver->get_dense_output());
            }
            
            // Handle events
            if (!events.empty()) {
                std::vector<double> g_new;
                g_new.reserve(events.size());
                for (const auto& event : events) {
                    g_new.push_back(event.function(t_target, result.y[i]));
                }
                
                auto active_events = detail::find_active_events(g_old, g_new, events);
                if (!active_events.empty()) {
                    if (!dense_output) {
                        solver->compute_dense_output();
                    }
                    
                    auto [event_indices, roots, terminate] = detail::handle_events(
                        solver->get_dense_output(),
                        events,
                        active_events,
                        t_current,
                        t_target
                    );
                    
                    // Store event times
                    for (size_t j = 0; j < event_indices.size(); ++j) {
                        result.t_events[event_indices[j]].push_back(roots[j]);
                    }
                    
                    if (terminate) {
                        result.t.resize(i + 1);
                        result.y.resize(i + 1);
                        result.status = 1;
                        result.message = "A termination event occurred.";
                        result.success = true;
                        
                        if (dense_output) {
                            // Include the last interpolant
                            interpolants.push_back(solver->get_dense_output());
                            result.sol = std::make_unique<OdeSolution>(
                                std::move(result.t),
                                std::move(interpolants)
                            );
                        }
                        
                        return result;
                    }
                }
                
                g_old = std::move(g_new);
            }
        }
        
        result.status = 0;
        result.message = "The solver successfully reached the end of the integration interval.";
        result.success = true;
        
        if (dense_output && !interpolants.empty()) {
            result.sol = std::make_unique<OdeSolution>(
                std::move(result.t),
                std::move(interpolants)
            );
        }
        
    } catch (const std::exception& e) {
        result.status = -1;
        result.message = std::string("Integration failed: ") + e.what();
        result.success = false;
    }
    
    return result;
}

} // namespace rk_solver