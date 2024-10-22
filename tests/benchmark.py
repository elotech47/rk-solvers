import numpy as np
import rk_solver_cpp
import time
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def benchmark(fun, y0, t_span, analytical_sol=None, name="", npoints=100, stiff=False):
    dt = (t_span[1] - t_span[0]) / npoints
    ts = np.linspace(t_span[0], t_span[1], npoints + 1)
    
    # C++ RK23 high tolerance
    print(f"y0: {y0}")
    print(f"t_span: {t_span}")
    rk23 = rk_solver_cpp.RK23(fun, t_span[0], y0, t_span[1], rtol=1e-10, atol=1e-10)
    start_time = time.time()
    ys_cpp = []
    try:
        for t in ts:
            rk23.integrate(t)
            ys_cpp.append(np.array(rk23.get_y()))
        time_cpp = time.time() - start_time
        cpp_success = True
    except Exception as e:
        print(f"C++ RK23 (high tol) failed: {str(e)}")
        time_cpp = None
        cpp_success = False
    
    # C++ RK45 low tolerance
    rk45_low = rk_solver_cpp.RK45(fun, t_span[0], y0, t_span[1])
    start_time = time.time()
    ys_cpp_low = []
    try:
        for t in ts:
            rk45_low.integrate(t)
            ys_cpp_low.append(np.array(rk45_low.get_y()))
        time_cpp_low = time.time() - start_time
        cpp_low_success = True
    except Exception as e:
        print(f"C++ RK45 (low tol) failed: {str(e)}")
        time_cpp_low = None
        cpp_low_success = False
    
    # SciPy solver
    start_time = time.time()
    sol_scipy = solve_ivp(fun, t_span, y0, method='RK23' if not stiff else 'Radau', 
                          t_eval=ts, rtol=1e-10, atol=1e-10)

    time_scipy = time.time() - start_time
    
    # scipy RK45
    start_time = time.time()
    sol_scipy_rk45 = solve_ivp(fun, t_span, y0, method='RK45', 
                               t_eval=ts, rtol=1e-6, atol=1e-8)
    time_scipy_rk45 = time.time() - start_time
    
    # Calculate errors if analytical solution is provided
    if analytical_sol:
        ys_analytical = analytical_sol(ts)
        error_cpp = np.max(np.abs(np.array(ys_cpp) - ys_analytical)) if cpp_success else None
        error_cpp_low = np.max(np.abs(np.array(ys_cpp_low) - ys_analytical)) if cpp_low_success else None
        error_scipy = np.max(np.abs(sol_scipy.y.T - ys_analytical))
        error_scipy_rk45 = np.max(np.abs(sol_scipy_rk45.y.T - ys_analytical))
    else:
        error_cpp = error_cpp_low = error_scipy = error_scipy_rk45 = "N/A"
    
    print(f"\nResults for {name}:")
    if cpp_success:
        print(f"C++ RK23 (high tol) time: {time_cpp:.6f} s, Max Error: {error_cpp}")
    if cpp_low_success:
        print(f"C++ RK45 time: {time_cpp_low:.6f} s, Max Error: {error_cpp_low}")
    print(f"SciPy Solver time: {time_scipy:.6f} s, Max Error: {error_scipy}")
    print(f"SciPy RK45 time: {time_scipy_rk45:.6f} s, Max Error: {error_scipy_rk45}")
    # Plotting
    plt.figure(figsize=(12, 6))
    if analytical_sol:
        plt.plot(ts, ys_analytical, label='Analytical', linestyle='-')
    if cpp_success:
        plt.plot(ts, ys_cpp, label='C++ RK23 (high tol)', linestyle='--', marker='.')
    if cpp_low_success:
        plt.plot(ts, ys_cpp_low, label='C++ RK45', linestyle=':', marker='.')
    plt.plot(sol_scipy.t, sol_scipy.y.T, label='SciPy', linestyle='-.', marker='.')
    plt.plot(sol_scipy_rk45.t, sol_scipy_rk45.y.T, label='SciPy RK45', linestyle=':', marker='.')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.title(f'{name} - Solution Comparison')
    plt.legend()
    plt.savefig(f'{name.lower().replace(" ", "_")}_comparison.png')
    plt.close()

# Test cases
problems = [
    {
        "name": "Simple Linear",
        "fun": lambda t, y: np.array([2*t]),
        "y0": np.array([0.0]),
        "t_span": (0.0, 5.0),
        "analytical_sol": lambda t: t**2
    },
    {
        "name": "Exponential Growth",
        "fun": lambda t, y: y,
        "y0": np.array([1.0]),
        "t_span": (0.0, 2.0),
        "analytical_sol": lambda t: np.exp(t)
    },
    {
        "name": "Harmonic Oscillator",
        "fun": lambda t, y: np.array([y[1], -y[0]]),
        "y0": np.array([0.0, 1.0]),
        "t_span": (0.0, 10.0),
        "analytical_sol": lambda t: np.array([np.sin(t), np.cos(t)]).T
    },
    {
        "name": "Nonlinear",
        "fun": lambda t, y: np.array([y[0]**2 - t**2]),
        "y0": np.array([0.0]),
        "t_span": (0.0, 2.0),
        "analytical_sol": lambda t: t * np.tan(t)
    },
    {
        "name": "Stiff Problem (Van der Pol)",
        "fun": lambda t, y: np.array([y[1], 1000*(1 - y[0]**2)*y[1] - y[0]]),
        "y0": np.array([2.0, 0.0]),
        "t_span": (0.0, 3000.0),
        "stiff": True
    }
]

for problem in problems:
    benchmark(**problem)