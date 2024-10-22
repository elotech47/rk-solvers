# import numpy as np
# from scipy.integrate import solve_ivp
# import rk_solver_cpp
# import matplotlib.pyplot as plt
# import time


# def test_exponential_decay():
#     def exponential_decay(t, y):
#         return [-0.5 * y[0]]  # Return a list with a single element

#     t_span = (0, 10)
#     y0 = [1.0]

#     rtol = 1e-6
#     atol = 1e-8
#     # # Solve using SciPy
#     start_time = time.time()
#     sol_scipy_rk23 = solve_ivp(exponential_decay, t_span, y0, method='RK23', rtol=rtol, atol=atol)
#     end_time = time.time()
#     print(f"Time taken by SciPy RK23: {end_time - start_time} seconds")
#     start_time = time.time()
#     sol_scipy_rk45 = solve_ivp(exponential_decay, t_span, y0, method='RK45', rtol=rtol, atol=atol)
#     end_time = time.time()
#     print(f"Time taken by SciPy RK45: {end_time - start_time} seconds")

#     # Solve using our C++ implementation
#     solver_cpp_rk23 = rk_solver_cpp.RK23(exponential_decay, t_span[0], y0, t_span[1], rtol=rtol, atol=atol)
#     solver_cpp_rk45 = rk_solver_cpp.RK45(exponential_decay, t_span[0], y0, t_span[1], rtol=rtol, atol=atol)

#     t_eval = np.linspace(t_span[0], t_span[1], 100)
#     y_cpp_rk23 = []
#     y_cpp_rk45 = []

#     rk23_time = 0
#     rk45_time = 0
#     for t in t_eval:
#         start_time = time.time()
#         solver_cpp_rk23.integrate(t)
#         end_time = time.time()
#         rk23_time += end_time - start_time
#         start_time = time.time()
#         solver_cpp_rk45.integrate(t)
#         end_time = time.time()
#         rk45_time += end_time - start_time
#         y_cpp_rk23.append(solver_cpp_rk23.get_y()[0])
#         y_cpp_rk45.append(solver_cpp_rk45.get_y()[0])
#         # print("----")

#     print(f"Time taken by C++ RK23: {rk23_time} seconds")
#     print(f"Time taken by C++ RK45: {rk45_time} seconds")

#     # Plot results
#     plt.figure(figsize=(12, 6))
#     plt.plot(sol_scipy_rk23.t, sol_scipy_rk23.y[0], 'b-', label='SciPy RK23')
#     plt.plot(sol_scipy_rk45.t, sol_scipy_rk45.y[0], 'r-', label='SciPy RK45')
#     plt.plot(t_eval, y_cpp_rk23, 'g--', label='C++ RK23')
#     plt.plot(t_eval, y_cpp_rk45, 'm--', label='C++ RK45')
#     plt.legend()
#     plt.title('Exponential Decay: SciPy vs C++ Implementation')
#     plt.xlabel('t')
#     plt.ylabel('y')
#     plt.savefig('exponential_decay_comparison.png')
#     plt.close()

#     # Compute errors
#     error_rk23 = np.abs(np.array(y_cpp_rk23) - np.interp(t_eval, sol_scipy_rk23.t, sol_scipy_rk23.y[0]))
#     error_rk45 = np.abs(np.array(y_cpp_rk45) - np.interp(t_eval, sol_scipy_rk45.t, sol_scipy_rk45.y[0]))

#     print(f"Max error RK23: {np.max(error_rk23)}")
#     print(f"Max error RK45: {np.max(error_rk45)}")

# def test_van_der_pol_oscillator():
#     def van_der_pol(t, y, mu=1.0):
#         return [y[1], mu * (1 - y[0]**2) * y[1] - y[0]]

#     t_span = (0, 20)
#     y0 = [2.0, 0.0]

#     # Solve using SciPy
#     start_time = time.time()
#     sol_scipy_rk45 = solve_ivp(van_der_pol, t_span, y0, method='RK45', dense_output=True)
#     end_time = time.time()
#     print(f"Time taken by SciPy RK45: {end_time - start_time} seconds")

#     # Solve using our C++ implementation
#     solver_cpp_rk45 = rk_solver_cpp.RK45(lambda t, y: van_der_pol(t, y), t_span[0], y0, t_span[1])

#     t_eval = np.linspace(t_span[0], t_span[1], 500)
#     y_cpp_rk45 = []
#     rk45_time = 0
#     for t in t_eval:
#         start_time = time.time()
#         solver_cpp_rk45.integrate(t)
#         end_time = time.time()
#         rk45_time += end_time - start_time
#         y_cpp_rk45.append(solver_cpp_rk45.get_y())

#     print(f"Time taken by C++ RK45: {rk45_time} seconds")

#     y_cpp_rk45 = np.array(y_cpp_rk45).T

#     # Plot results
#     plt.figure(figsize=(12, 6))
#     plt.plot(sol_scipy_rk45.t, sol_scipy_rk45.y[0], 'b-', label='SciPy RK45 (x)')
#     plt.plot(sol_scipy_rk45.t, sol_scipy_rk45.y[1], 'r-', label='SciPy RK45 (y)')
#     plt.plot(t_eval, y_cpp_rk45[0], 'g--', label='C++ RK45 (x)')
#     plt.plot(t_eval, y_cpp_rk45[1], 'm--', label='C++ RK45 (y)')
#     plt.legend()
#     plt.title('Van der Pol Oscillator: SciPy vs C++ Implementation')
#     plt.xlabel('t')
#     plt.ylabel('x, y')
#     plt.savefig('van_der_pol_comparison.png')
#     plt.close()

#     # Compute errors
#     error_x = np.abs(y_cpp_rk45[0] - sol_scipy_rk45.sol(t_eval)[0])
#     error_y = np.abs(y_cpp_rk45[1] - sol_scipy_rk45.sol(t_eval)[1])

#     print(f"Max error x: {np.max(error_x)}")
#     print(f"Max error y: {np.max(error_y)}")

# if __name__ == "__main__":
#     test_exponential_decay()
#     test_van_der_pol_oscillator()


import numpy as np
import rk_solver_cpp
import time
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


f_cpp = lambda t, y: np.array([2*t])  # Ensure the function returns a NumPy array
analytical_solution = lambda t: t**2
y0 = np.array([0.0])  # Use a NumPy array for y0
ts_cpp = []
ts_cpp_low = []
ys_cpp = []
ys_cpp_low = []
t_span = (0.0, 5.0)
npoints = 100
dt = t_span[1]/npoints       
rk23 = rk_solver_cpp.RK23(f_cpp, t_span[0], y0, t_span[1], rtol=1e-10, atol=1e-10)
rk23_low = rk_solver_cpp.RK23(f_cpp, t_span[0], y0, t_span[1], rtol=1e-6, atol=1e-8)

total_time = 0
total_time_low = 0
for i in range(npoints):
    t1 = t_span[0] + (i+1)*dt
    start_time = time.time()
    rk23.integrate(t1)
    total_time += time.time() - start_time
    y_sol = np.array(rk23.get_y())
    ts_cpp.append(t1)
    ys_cpp.append(y_sol)
    start_time = time.time()
    rk23_low.integrate(t1)
    total_time_low += time.time() - start_time
    y_sol = np.array(rk23_low.get_y())
    ts_cpp_low.append(t1)
    ys_cpp_low.append(y_sol)
 # Analytical solution
ts_analytical = np.linspace(t_span[0], t_span[1], npoints + 1)
ys_analytical = analytical_solution(ts_analytical)

start_scipy = time.time()
sol_scipy = solve_ivp(f_cpp, t_span, y0, method='RK23', t_eval=np.linspace(t_span[0], t_span[1], npoints + 1))
time_scipy = time.time() - start_scipy
print(f"SciPy Solver time: {time_scipy:.6f} seconds")
print(f"Total time taken by C++ RK23: {total_time:.6f} seconds")
print(f"Total time taken by C++ RK23 low: {total_time_low:.6f} seconds")

# Plotting the results
plt.figure(figsize=(12, 6))
plt.plot(ts_analytical, ys_analytical, label='Analytical Solution', linestyle='-', marker='o')
plt.plot(ts_cpp, ys_cpp, label='C++ RK23 Solver', linestyle='-', marker='o')
plt.plot(ts_cpp_low, ys_cpp_low, label='C++ RK23 Solver Low Tolerance', linestyle='-', marker='.')
plt.plot(sol_scipy.t, sol_scipy.y.T, label='SciPy Solver', linestyle='-', marker='*')
plt.xlabel('t')
plt.ylabel('y')
plt.legend()
plt.savefig('rk23_comparison.png')
plt.close()
