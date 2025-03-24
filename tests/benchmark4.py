import numpy as np
import matplotlib.pyplot as plt
import cantera as ct
import time
import rk_solver_cpp
from scipy.integrate import ode
from tqdm import tqdm


def setup_gas(T, pp):
    gas = ct.Solution('RL/ppo/mechanism_files/ch4_53species.yaml')
    gas.TP = T, ct.one_atm*pp
    gas.set_equivalence_ratio(1.0, 'CH4', 'O2:1.0, N2:3.76')
    return gas, pp 

def combustion_ode(gas, pp):
    def f(t, Y):  # For cpp solver
        T = Y[0]
        YY = Y[1:]
        gas.TPY = T, ct.one_atm*pp, YY
        species_rates = gas.net_production_rates*gas.molecular_weights/gas.density
        species_h = gas.partial_molar_enthalpies/gas.molecular_weights
        temp_rate = -np.sum(species_rates*species_h/gas.cp_mass)
        return np.concatenate((np.array([temp_rate]), species_rates), axis=0)
    
    def f_ode(t, Y):  # For scipy.integrate.ode
        T = Y[0]
        YY = Y[1:]
        gas.TPY = T, ct.one_atm*pp, YY
        species_rates = gas.net_production_rates*gas.molecular_weights/gas.density
        species_h = gas.partial_molar_enthalpies/gas.molecular_weights
        temp_rate = -np.sum(species_rates*species_h/gas.cp_mass)
        return np.concatenate((np.array([temp_rate]), species_rates), axis=0)
    
    return f, f_ode

def benchmark_combustion(gas, pp, t_span, npoints, integrator='dopri5', method=None):
    f, f_ode = combustion_ode(gas, pp)
    y0 = np.hstack([[gas.T], gas.Y])
    ts = np.linspace(t_span[0], t_span[1], npoints + 1)
    dt = t_span[1]/npoints
    
    r = ode(f_ode)
    r.set_integrator(integrator, rtol=1e-6, atol=1e-8, method=method)
    r.set_initial_value(y0, t_span[0])
    
      # SciPy ode
    print(f"Solving with SciPy {integrator}")
    start_time = time.time()
    # Integration loop
    ys_scipy = []
    ts_scipy = []
    scipy_success = True
    
    try:
        for t in ts:
            r.integrate(t)
            ys_scipy.append(r.y)
            ts_scipy.append(r.t)
        time_scipy = time.time() - start_time
    except Exception as e:
        print(f"SciPy {integrator} integration failed: {str(e)}")
        scipy_success = False
    
    if scipy_success:
        ys_scipy = np.array(ys_scipy).T
        
    print(f"SciPy {integrator} time: {time_scipy:.6f} s")
        
    # C++ RK23
    print(f"Solving with C++ RK23")
    rk23 = rk_solver_cpp.RK23(f, t_span[0], y0, t_span[1], rtol=1e-10, atol=1e-10)
    
    ts_cpp = []
    ys_cpp = []
    total_time = 0
    cpp_success = True

    try:
        for i in range(npoints):
            t1 = t_span[0] + (i+1)*dt
            
            # High precision RK23
            start_time = time.time()
            rk23.integrate(t1)
            total_time += time.time() - start_time
            y_sol = np.array(rk23.get_y())
            ts_cpp.append(t1)
            ys_cpp.append(y_sol)
            
    except Exception as e:
        print(f"C++ RK23 failed: {str(e)}")
        cpp_success = False

    # Convert lists to arrays for plotting
    if cpp_success:
        ys_cpp = np.array(ys_cpp).T


    # Print results
    print(f"\nResults for Combustion Integration:")
    print(f"C++ RK23 (high precision) time: {total_time:.6f} s")
    print(f"SciPy {integrator} time: {time_scipy:.6f} s")

    # Plotting temperature
    plt.figure(figsize=(12, 6))
    if cpp_success:
        plt.plot(ts_cpp, ys_cpp[0], label='C++ RK23 (high precision)', linestyle='--')
    if scipy_success:
        plt.plot(ts_scipy, ys_scipy[0], label=f'SciPy {integrator}', linestyle='-')
    plt.xlabel('Time (s)')
    plt.ylabel('Temperature (K)')
    plt.title('Combustion Integration - Temperature vs Time')
    plt.legend()
    plt.savefig(f'combustion_integration_comparison_{integrator}.png')
    plt.close()

    # Plot species mass fractions
    species_to_plot = ['CH4', 'O2', 'CO2', 'H2O']
    species_indices = [gas.species_index(s) for s in species_to_plot]
    
    plt.figure(figsize=(12, 8))
    for i, species in enumerate(species_to_plot):
        if cpp_success:
            plt.plot(ts_cpp, ys_cpp[species_indices[i]+1], 
                    label=f'{species} (C++ RK23 high)', linestyle='--')
        if scipy_success:
            plt.plot(ts_scipy, ys_scipy[species_indices[i]+1], 
                    label=f'{species} (SciPy {integrator})', linestyle='-')
    plt.xlabel('Time (s)')
    plt.ylabel('Mass Fraction')
    plt.title(f'Combustion Integration - Species Mass Fractions ({integrator})')
    plt.legend()
    plt.yscale('log')
    plt.savefig(f'combustion_integration_species_comparison_{integrator}.png')
    plt.close()
    
    
    # compare the time taken for each step

if __name__ == "__main__":
    T = 1400
    pp = 40
    gas, pp = setup_gas(T, pp)
    t_span = (0.0, 2e-4)
    npoints = 100
    
    # List of integrators to try
    integrator = 'vode'
    method = 'adams'
    benchmark_combustion(gas, pp, t_span, npoints, integrator=integrator, method=method)
