import numpy as np
import rk_solver_cpp
import time
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def benchmark(fun, y0, t_span, analytical_sol=None, name="", npoints=100, stiff=False):
    dt = (t_span[1] - t_span[0]) / npoints
    ts = np.linspace(t_span[0], t_span[1], npoints + 1)
    
    # C++ RK23 high tolerance
    print(f"Testing {name}")
    print(f"y0: {y0}")
    print(f"t_span: {t_span}")
    
    # Create RK23 solver and use solve_ivp
    start_time = time.time()
    try:
        rk23 = rk_solver_cpp.RK23(fun, t_span[0], y0, t_span[1], rtol=1e-10, atol=1e-10)
        result = rk_solver_cpp.solve_ivp(rk23, ts)
        ys_cpp = np.array(result['y'])
        time_cpp = time.time() - start_time
        cpp_success = result['success']
        if not cpp_success:
            print(f"C++ RK23 (high tol) warning: {result['message']}")
    except Exception as e:
        print(f"C++ RK23 (high tol) failed: {str(e)}")
        time_cpp = None
        cpp_success = False
    
    # C++ RK45 low tolerance
    start_time = time.time()
    try:
        rk45 = rk_solver_cpp.RK45(fun, t_span[0], y0, t_span[1])
        result_low = rk_solver_cpp.solve_ivp(rk45, ts)
        ys_cpp_low = np.array(result_low['y'])
        time_cpp_low = time.time() - start_time
        cpp_low_success = result_low['success']
        if not cpp_low_success:
            print(f"C++ RK45 warning: {result_low['message']}")
    except Exception as e:
        print(f"C++ RK45 failed: {str(e)}")
        time_cpp_low = None
        cpp_low_success = False
    
    # SciPy solver for comparison
    start_time = time.time()
    sol_scipy = solve_ivp(fun, t_span, y0, method='RK23' if not stiff else 'Radau', 
                         t_eval=ts, rtol=1e-10, atol=1e-10)
    time_scipy = time.time() - start_time
    
    # Calculate errors if analytical solution is provided
    if analytical_sol is not None:
        ys_analytical = np.array([analytical_sol(t) for t in ts])
        if cpp_success:
            error_cpp = np.max(np.abs(ys_cpp - ys_analytical))
        else:
            error_cpp = None
        if cpp_low_success:
            error_cpp_low = np.max(np.abs(ys_cpp_low - ys_analytical))
        else:
            error_cpp_low = None
        error_scipy = np.max(np.abs(sol_scipy.y.T - ys_analytical))
    else:
        error_cpp = error_cpp_low = error_scipy = "N/A"
    
    # Print results
    print(f"\nResults for {name}:")
    if cpp_success:
        print(f"C++ RK23 (high tol) time: {time_cpp:.6f} s, Max Error: {error_cpp}")
    if cpp_low_success:
        print(f"C++ RK45 time: {time_cpp_low:.6f} s, Max Error: {error_cpp_low}")
    print(f"SciPy Solver time: {time_scipy:.6f} s, Max Error: {error_scipy}")
    
    # Plotting
    plt.figure(figsize=(12, 6))
    if analytical_sol is not None:
        plt.plot(ts, ys_analytical, label='Analytical', linestyle='-')
    if cpp_success:
        plt.plot(ts, ys_cpp, label='C++ RK23 (high tol)', linestyle='--', marker='.')
    if cpp_low_success:
        plt.plot(ts, ys_cpp_low, label='C++ RK45', linestyle=':', marker='.')
    plt.plot(sol_scipy.t, sol_scipy.y.T, label='SciPy', linestyle='-.', marker='.')
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
        "analytical_sol": lambda t: np.array([t**2])
    },
    {
        "name": "Exponential Growth",
        "fun": lambda t, y: y,
        "y0": np.array([1.0]),
        "t_span": (0.0, 2.0),
        "analytical_sol": lambda t: np.array([np.exp(t)])
    },
    {
        "name": "Harmonic Oscillator",
        "fun": lambda t, y: np.array([y[1], -y[0]]),
        "y0": np.array([0.0, 1.0]),
        "t_span": (0.0, 10.0),
        "analytical_sol": lambda t: np.array([np.sin(t), np.cos(t)])
    }
]

# Run benchmarks
for problem in problems:
    benchmark(**problem)