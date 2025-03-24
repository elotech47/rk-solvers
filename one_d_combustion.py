import cantera as ct
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from tqdm import tqdm


class Flame1D:
    def __init__(self, mech='gri30.yaml', fuel='CH4', phi=1.0, T_inlet=300.0, P=ct.one_atm):
        # Initialize gas object
        self.gas = ct.Solution(mech)
        
        # Set inlet conditions
        self.gas.set_equivalence_ratio(phi, fuel, {'O2': 1.0, 'N2': 3.76})
        self.gas.TPX = T_inlet, P, self.gas.X
        
        # Store parameters
        self.T_inlet = T_inlet
        self.P = P
        self.n_species = self.gas.n_species
        
        # Transport properties
        self.gas.transport_model = 'mixture-averaged'
        
    def equations_ivp(self, t, y):
        """System of ODEs for time evolution"""
        # Reshape y into temperature and species profiles
        n_points = len(y) // (self.n_species + 1)
        T = y[:n_points]
        Y = y[n_points:].reshape(self.n_species, n_points)
        
        # Spatial grid (fixed)
        dx = self.width / (n_points - 1)
        x = np.linspace(0, self.width, n_points)
        
        # Initialize derivative arrays
        dTdt = np.zeros_like(T)
        dYdt = np.zeros_like(Y)
        
        # Calculate spatial derivatives
        for i in tqdm(range(1, n_points-1), desc="Calculating spatial derivatives"):
            # Update gas state
            self.gas.TPY = T[i], self.P, Y[:, i]
            
            # Transport properties
            k = self.gas.thermal_conductivity
            D = self.gas.mix_diff_coeffs
            rho = self.gas.density
            cp = self.gas.cp_mass
            
            # Temperature diffusion term
            d2Tdx2 = (T[i+1] - 2*T[i] + T[i-1]) / dx**2
            dTdt[i] = k/(rho*cp) * d2Tdx2
            
            # Species diffusion and reaction terms
            for j in range(self.n_species):
                d2Ydx2 = (Y[j,i+1] - 2*Y[j,i] + Y[j,i-1]) / dx**2
                dYdt[j,i] = D[j]/rho * d2Ydx2
            
            # Add reaction source terms
            net_production = self.gas.net_production_rates
            dYdt[:,i] += net_production * self.gas.molecular_weights / rho
            
            # Add heat release term
            Q = -np.sum(net_production * self.gas.partial_molar_enthalpies)
            dTdt[i] += Q / (rho * cp)
        
        # Apply boundary conditions
        dTdt[0] = 0  # Fixed inlet temperature
        dTdt[-1] = 0  # Zero gradient at outlet
        dYdt[:,0] = 0  # Fixed inlet composition
        dYdt[:,-1] = 0  # Zero gradient at outlet
        
        return np.concatenate([dTdt, dYdt.flatten()])
    
    def solve_ivp(self, width, n_points, t_span, dt_save=0.001):
        """Solve the time evolution"""
        self.width = width
        x = np.linspace(0, width, n_points)
        
        # Initial conditions: create a hot spot in the middle
        T_init = self.T_inlet + 1500 * np.exp(-((x - width/2)/(width/10))**2)
        Y_init = np.tile(self.gas.Y, (n_points, 1)).T
        y0 = np.concatenate([T_init, Y_init.flatten()])
        
        # Time points to save
        t_eval = np.arange(t_span[0], t_span[1], dt_save)
        
        # Solve
        sol = solve_ivp(
            self.equations_ivp,
            t_span,
            y0,
            method='BDF',
            t_eval=t_eval,
            rtol=1e-4,
            atol=1e-6
        )
        
        if sol.success:
            print("Solution converged!")
            return sol, x
        else:
            print("Solution failed to converge")
            return None, None

def animate_temperature(sol, x, save_animation=False):
    """Create animation of temperature evolution"""
    if sol is None:
        return None
    
    print("Creating animation")
    # Extract temperature profiles
    n_points = len(x)
    T = sol.y[:n_points, :]
    
    # Create animation
    fig, ax = plt.subplots(figsize=(10, 6))
    line, = ax.plot([], [], 'r-', lw=2)
    
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(np.min(T), np.max(T))
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Temperature (K)')
    ax.grid(True)
    
    def init():
        line.set_data([], [])
        return line,
    
    def animate(frame):
        line.set_data(x, T[:, frame])
        ax.set_title(f'Time: {sol.t[frame]:.3f} s')
        return line,
    
    anim = FuncAnimation(fig, animate, init_func=init,
                        frames=len(sol.t), interval=50,
                        blit=True)
    
    if save_animation:
        anim.save('temperature_evolution.gif', writer='pillow')
    
    plt.close()
    return HTML(anim.to_jshtml())

def plot_results(sol, x):
    """Plot final temperature profile and evolution at specific points"""
    print("Plotting results")
    if sol is None:
        return
    
    n_points = len(x)
    T = sol.y[:n_points, :]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Final temperature profile
    ax1.plot(x * 100, T[:, -1], 'r-', label='Final')
    ax1.plot(x * 100, T[:, 0], 'b--', label='Initial')
    ax1.set_xlabel('Distance (cm)')
    ax1.set_ylabel('Temperature (K)')
    ax1.grid(True)
    ax1.legend()
    
    # Temperature evolution at specific points
    points = [int(n_points*0.25), int(n_points*0.5), int(n_points*0.75)]
    for p in points:
        ax2.plot(sol.t, T[p, :], label=f'x = {x[p]*100:.1f} cm')
    
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Temperature (K)')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    return fig

# Example usage
if __name__ == "__main__":
    # Create and solve
    flame = Flame1D()
    sol, x = flame.solve_ivp(
        width=0.01,      # 2 cm domain
        n_points=10,    # spatial resolution
        t_span=[0, 0.01],  # simulate for 0.1 seconds
        dt_save=0.0001
    )
    
    # Create static plots
    fig = plot_results(sol, x)
    plt.show()
    
    # Create animation
    anim = animate_temperature(sol, x, save_animation=True)
