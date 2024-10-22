import numpy as np
import matplotlib.pyplot as plt
import cantera as ct
import time
import rk_solver_cpp
from scipy.integrate import solve_ivp
from tqdm import tqdm


def setup_gas(T, pp):
    gas = ct.Solution('mechanism_files/ch4_53species.yaml')
    gas.TP = T, ct.one_atm*pp
    gas.set_equivalence_ratio(1.0, 'CH4', 'O2:1.0, N2:3.76')
    return gas, pp 

def combustion_ode(gas, pp):
    def f(t, Y):
        T = Y[0]
        YY = Y[1:]
        gas.TPY = T, ct.one_atm*pp, YY
        species_rates = gas.net_production_rates*gas.molecular_weights/gas.density
        species_h = gas.partial_molar_enthalpies/gas.molecular_weights
        temp_rate = -np.sum(species_rates*species_h/gas.cp_mass)
        return np.concatenate((np.array([temp_rate]), species_rates), axis=0)
    return f


def benchmark_combustion(gas, pp, t_span, npoints):
    f = combustion_ode(gas, pp)
    y0 = np.hstack([[gas.T], gas.Y])
    ts = np.linspace(t_span[0], t_span[1], npoints + 1)

    # C++ RK23
    rk23 = rk_solver_cpp.RK23(f, t_span[0], y0, t_span[1], rtol=1e-6, atol=1e-8)
    print(f"Solving with C++ RK23")
    start_time = time.time()
    try:
        result_cpp = rk_solver_cpp.solve_ivp(rk23, ts)
        time_cpp = time.time() - start_time
        cpp_success = result_cpp['success']
        if cpp_success:
            ys_cpp = result_cpp['y']
        else:
            print(f"C++ RK23 failed: {result_cpp['message']}")
    except Exception as e:
        print(f"C++ RK23 failed: {str(e)}")
        time_cpp = None
        cpp_success = False

    # SciPy RK23
    print(f"Solving with SciPy RK23")
    start_time = time.time()
    
    sol_scipy = solve_ivp(f, t_span, y0, method='RK23', t_eval=ts, rtol=1e-6, atol=1e-8)
    time_scipy = time.time() - start_time

    # Print results
    print(f"\nResults for Combustion Integration:")
    if cpp_success:
        print(f"C++ RK23 time: {time_cpp:.6f} s")
    print(f"SciPy RK23 time: {time_scipy:.6f} s")

    # Plotting
    plt.figure(figsize=(12, 6))
    if cpp_success:
        plt.plot(ts, [y[0] for y in ys_cpp], label='C++ RK23', linestyle='--')
    plt.plot(sol_scipy.t, sol_scipy.y[0], label='SciPy RK23', linestyle='-')
    plt.xlabel('Time (s)')
    plt.ylabel('Temperature (K)')
    plt.title('Combustion Integration - Temperature vs Time')
    plt.legend()
    plt.savefig('combustion_integration_comparison.png')
    plt.close()

    # Plot species mass fractions
    species_to_plot = ['CH4', 'O2', 'CO2', 'H2O']
    species_indices = [gas.species_index(s) for s in species_to_plot]
    
    plt.figure(figsize=(12, 8))
    for i, species in enumerate(species_to_plot):
        if cpp_success:
            plt.plot(ts, [y[species_indices[i]+1] for y in ys_cpp], label=f'{species} (C++ RK23)', linestyle='--')
        plt.plot(sol_scipy.t, sol_scipy.y[species_indices[i]+1], label=f'{species} (SciPy RK23)', linestyle='-')
    plt.xlabel('Time (s)')
    plt.ylabel('Mass Fraction')
    plt.title('Combustion Integration - Species Mass Fractions')
    plt.legend()
    plt.yscale('log')
    plt.savefig('combustion_integration_species_comparison.png')
    plt.close()

if __name__ == "__main__":
    T = 1400
    pp = 40
    gas, pp = setup_gas(T, pp)
    t_span = (0.0, 2e-4)
    npoints = 100
    benchmark_combustion(gas, pp, t_span, npoints)