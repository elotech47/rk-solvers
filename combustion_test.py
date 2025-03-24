import numpy as np
import cantera as ct
from scipy.integrate import solve_ivp
import torch
from multiprocessing import Pool, cpu_count
from functools import partial
import time
from collections import deque
import matplotlib.pyplot as plt

class PolicyGuidedSolverND:
    def __init__(self, 
                 policy_model,
                 mech_file, 
                 initial_state, 
                 dimensions=(100,),
                 domain_size=(0.02,),
                 end_time=1e-3,
                 dt=1e-5,
                 species_to_track=['H2', 'O2', 'H', 'OH', 'H2O'],
                 Etol=1e-7,
                 features_config={
                     'temporal_features': True,
                     'species_features': True,
                     'stiffness_features': False,
                     'basic_features': True,
                     'include_dt_etol': False,
                     'add_net_production_rates': False
                 },
                 window_size=5,
                 feature_params={
                     'epsilon': 1e-10,
                     'clip_value': 1e-20
                 },
                 integrator_list=['CPP_RK23', 'Radau', 'BDF'],
                 tolerance_list=[(1e-12, 1e-14), (1e-6, 1e-8)],
                 batch_size=1000,
                 n_processes=None):
        
        self.policy = policy_model
        self.dimensions = dimensions
        self.domain_size = domain_size
        self.n_dim = len(dimensions)
        self.dt = dt
        self.end_time = end_time
        self.species_to_track = species_to_track
        self.Etol = Etol
        self.features_config = features_config
        self.window_size = window_size
        self.feature_params = feature_params
        self.integrator_list = integrator_list
        self.tolerance_list = tolerance_list
        self.batch_size = batch_size
        
        # Set up parallel processing
        self.n_processes = n_processes or max(1, cpu_count() - 1)
        
        # Initialize Cantera
        self.gas = ct.Solution(mech_file)
        self.gas.TPX = initial_state['T'], initial_state['P'], initial_state['X']
        self.n_species = self.gas.n_species
        self.P = initial_state['P']
        
        # Create spatial grids
        self.grids = []
        self.dx = []
        for dim, size in zip(dimensions, domain_size):
            grid = np.linspace(0, size, dim)
            self.grids.append(grid)
            self.dx.append(size / (dim - 1))
        
        # Initialize solution arrays
        self.grid_shape = tuple(dimensions)
        self.T = np.ones(self.grid_shape) * initial_state['T']
        self.Y = np.tile(self.gas.Y, (*self.grid_shape, 1))
        
        # Initialize history buffers for features
        self.history_buffer = self._initialize_feature_components()
        
        # Caching for similar states
        self.integrator_cache = {}
        self.cache_tolerance = 1e-6
        
        # Performance tracking
        self.performance_metrics = {
            'cpu_times': [],
            'errors': [],
            'solver_selections': [],
            'parallel_overhead': [],
            'cache_hits': [],
            'unique_states': []
        }
        
    def _initialize_feature_components(self):
        """Initialize components needed for selected features"""
        history_buffer = {}
        
        if self.features_config.get('temporal_features', False):
            history_buffer['temperature'] = deque(maxlen=self.window_size)
            history_buffer['gradients'] = deque(maxlen=self.window_size)
            
        if self.features_config.get('species_features', False):
            history_buffer['species'] = {
                spec: deque(maxlen=self.window_size) 
                for spec in self.species_to_track
            }
            
        return history_buffer
    
    def _update_history(self, T, Y, point_idx):
        """Update history buffers for a specific point"""
        if self.features_config.get('temporal_features', False):
            if len(self.history_buffer['temperature']) == self.window_size:
                self.history_buffer['temperature'].popleft()
                self.history_buffer['gradients'].popleft()
            
            self.history_buffer['temperature'].append(T)
            if len(self.history_buffer['temperature']) > 1:
                dT_dt = (self.history_buffer['temperature'][-1] - 
                        self.history_buffer['temperature'][-2]) / self.dt
            else:
                dT_dt = 0.0
            self.history_buffer['gradients'].append(dT_dt)
            
        if self.features_config.get('species_features', False):
            for i, spec in enumerate(self.species_to_track):
                if len(self.history_buffer['species'][spec]) == self.window_size:
                    self.history_buffer['species'][spec].popleft()
                self.history_buffer['species'][spec].append(Y[i])
    
    def _compute_temporal_features(self):
        """Compute temporal features if enabled"""
        if not self.features_config.get('temporal_features', False):
            return np.array([])
            
        if len(self.history_buffer['temperature']) < 2:
            return np.zeros(4)
            
        temp_array = np.array(self.history_buffer['temperature'])
        dT_dt = np.gradient(temp_array) / self.dt
        d2T_dt2 = np.gradient(dT_dt) / self.dt
        
        max_rate = np.max(np.abs(dT_dt))
        mean_rate = np.mean(np.abs(dT_dt))
        rate_variability = np.std(dT_dt)
        acceleration = np.mean(np.abs(d2T_dt2))
        
        return np.array([
            np.log1p(max_rate),
            np.log1p(mean_rate),
            np.log1p(rate_variability),
            np.log1p(acceleration)
        ])
    
    def _compute_species_features(self):
        """Compute species features if enabled"""
        if not self.features_config.get('species_features', False):
            return np.array([])
            
        features = []
        for spec in self.species_to_track:
            if len(self.history_buffer['species'][spec]) < 2:
                features.extend([0.0, 0.0, 0.0])
                continue
                
            spec_array = np.array(self.history_buffer['species'][spec])
            dY_dt = np.gradient(spec_array) / self.dt
            
            features.extend([
                np.log1p(np.max(np.abs(dY_dt))),
                np.log1p(np.mean(np.abs(dY_dt))),
                np.log1p(np.std(dY_dt))
            ])
            
        return np.array(features)
    
    def _compute_stiffness_features(self):
        """Compute stiffness features if enabled"""
        if not self.features_config.get('stiffness_features', False):
            return np.array([])
            
        epsilon = self.feature_params['epsilon']
        
        # Compute rate ratios
        net_rates = self.gas.net_production_rates
        significant_rates = np.abs(net_rates[np.abs(net_rates) > epsilon])
        
        if len(significant_rates) > 0:
            rate_ratio = np.max(significant_rates) / (np.min(significant_rates) + epsilon)
        else:
            rate_ratio = 1.0
            
        # Temperature gradient
        dT_dt = (self.history_buffer['gradients'][-1] 
                if len(self.history_buffer['gradients']) > 0 
                else 0.0)
        
        # Stiffness ratio from eigenvalues
        eigenvals = np.linalg.eigvals(self.gas.jacobian)
        stiffness_ratio = np.max(np.abs(eigenvals.real)) / (np.min(np.abs(eigenvals.real)) + epsilon)
        
        return np.array([
            np.log1p(rate_ratio),
            np.log1p(np.abs(dT_dt)),
            np.log1p(stiffness_ratio)
        ])

    def get_observation(self, T, Y, point_idx):
        """Get observation for policy based on local state"""
        self._update_history(T, Y, point_idx)
        
        observation_parts = []
        
        # Basic features
        if self.features_config.get('basic_features', True):
            self.gas.TPY = T, self.P, Y
            
            # Get species mass fractions for tracked species
            Y_tracked = np.array([Y[self.gas.species_index(spec)] 
                                for spec in self.species_to_track])
            
            if self.features_config.get('add_net_production_rates', False):
                net_rates = np.array([self.gas.net_production_rates[self.gas.species_index(spec)] 
                                    for spec in self.species_to_track])
                Y_tracked = np.hstack([Y_tracked, net_rates])
            
            # Process species data
            Y_tracked = np.clip(Y_tracked, self.feature_params['clip_value'], None)
            Y_log = np.log(Y_tracked)
            Y_normalized = (Y_log - np.mean(Y_log)) / np.std(Y_log)
            
            # Temperature normalization
            T_normalized = T / self.gas.T
            
            observation_parts.extend([Y_normalized, [T_normalized]])
        
        # Additional features
        observation_parts.extend([
            self._compute_temporal_features(),
            self._compute_species_features(),
            self._compute_stiffness_features()
        ])
        
        if self.features_config.get('include_dt_etol', False):
            dt_log = np.log(self.dt)
            Etol_log = np.log(self.Etol)
            observation_parts.extend([[dt_log, Etol_log]])
        
        return np.hstack(observation_parts).astype(np.float32)

    def _cache_key(self, T, Y):
        """Create cache key from rounded state values"""
        T_key = round(T / self.cache_tolerance) * self.cache_tolerance
        Y_key = tuple(round(y / self.cache_tolerance) * self.cache_tolerance 
                     for y in Y)
        return (T_key, Y_key)

    def batch_select_integrators(self, states_batch):
        """Process multiple states at once for efficiency"""
        observations = torch.stack([
            torch.from_numpy(self.get_observation(T, Y, i)) 
            for i, (T, Y) in enumerate(states_batch)
        ])
        
        # Batch process through policy
        with torch.no_grad():
            actions = self.policy(observations).argmax(dim=1)
            
        return [
            (self.integrator_list[a.item()], 
             *self.tolerance_list[a.item() % len(self.tolerance_list)])
            for a in actions
        ]

    def compute_spatial_derivatives(self, field):
        """Compute spatial derivatives for any field (T or Y)"""
        derivatives = np.zeros_like(field)
        second_derivatives = np.zeros_like(field)
        
        for dim in range(self.n_dim):
            slices = [slice(None)] * self.n_dim
            derivatives_dim = np.gradient(field, self.dx[dim], axis=dim)
            second_derivatives_dim = np.gradient(derivatives_dim, self.dx[dim], axis=dim)
            second_derivatives += second_derivatives_dim
            
        return second_derivatives

    def _solve_point(self, args):
        """Solve ODE for a single spatial point"""
        idx, t, dt, T, Y, d2Tdx2, d2Ydx2, integrator_choice = args
        integrator, rtol, atol = integrator_choice
        
        try:
            # Local ODE system
            def local_ode(t, y):
                T_local = y[0]
                Y_local = y[1:]
                
                # Update gas state
                self.gas.TPY = T_local, self.P, Y_local
                
                # Get properties
                wdot = self.gas.net_production_rates
                h = self.gas.partial_molar_enthalpies
                cp = self.gas.cp_mass
                rho = self.gas.density
                k = self.gas.thermal_conductivity
                D = self.gas.mix_diff_coeffs
                
                # Temperature equation
                dTdt = (k * d2Tdx2 - np.dot(h, wdot)) / (rho * cp)
                
                # Species equations
                dYdt = wdot * self.gas.molecular_weights / rho + D * d2Ydx2
                
                return np.hstack([dTdt, dYdt])
            
            # Solve
            y0 = np.hstack([T, Y])
            sol = solve_ivp(local_ode, (t, t+dt), y0, 
                          method=integrator, rtol=rtol, atol=atol)
            
            success = sol.success
            result = sol.y[:,-1] if success else y0
            action = self.integrator_list.index(integrator)
            
            return idx, result, action, success
            
        except Exception as e:
            print(f"Error at point {idx}: {e}")
            return idx, np.hstack([T, Y]), 0, False

    def solve_step(self, t):
        """Solve one time step using parallel processing"""
        start_time = time.time()
        
        # Compute spatial derivatives
        d2Tdx2 = self.compute_spatial_derivatives(self.T)
        d2Ydx2 = self.compute_spatial_derivatives(self.Y)
        
        # Prepare all points
        flat_indices = np.arange(np.prod(self.grid_shape))
        states = []
        cache_hits = 0
        unique_states = []
        
        # Check cache and collect unique states
        for idx in flat_indices:
            multi_idx = np.unravel_index(idx, self.grid_shape)
            T_local = self.T[multi_idx]
            Y_local = self.Y[multi_idx]
            
            cache_key = self._cache_key(T_local, Y_local)
            if cache_key in self.integrator_cache:
                cache_hits += 1
            else:
                unique_states.append((T_local, Y_local))
        
        # Process unique states in batches
        integrator_choices = {}
        for i in range(0, len(unique_states), self.batch_size):
            batch = unique_states[i:i + self.batch_size]
            choices = self.batch_select_integrators(batch)
            
            for (T, Y), choice in zip(batch, choices):
                cache_key = self._cache_key(T, Y)
                self.integrator_cache[cache_key] = choice
                integrator_choices[cache_key] = choice
        
        # Parallel processing with cached choices
        with Pool(processes=self.n_processes) as pool:
            args_list = []
            for idx in flat_indices:
                multi_idx = np.unravel_index(idx, self.grid_shape)
                T_local = self.T[multi_idx]
                Y_local = self.Y[multi_idx]
                d2T_local = d2Tdx2[multi_idx]
                d2Y_local = d2Ydx2[multi_idx]
                
                cache_key = self._cache_key(T_local, Y_local)
                integrator_choice = self.integrator_cache[cache_key]
                
                args_list.append((
                    idx, t, self.dt, T_local, Y_local, 
                    d2T_local, d2Y_local, integrator_choice
                ))
            
            results = pool.map(self._solve_point, args_list)
        
        # Process results
        T_new = np.zeros_like(self.T)
        Y_new = np.zeros_like(self.Y)
        actions = np.zeros(self.grid_shape, dtype=int)
        
        for idx, result, action, success in results:
            if success:
                multi_idx = np.unravel_index(idx, self.grid_shape)
                T_new[multi_idx] = result[0]
                Y_new[multi_idx] = result[1:]
                actions[multi_idx] = action
        
        # Update solution
        self.T = T_new
        self.Y = Y_new
        
        # Update performance metrics
        parallel_time = time.time() - start_time
        self.performance_metrics['parallel_overhead'].append(parallel_time)
        self.performance_metrics['solver_selections'].append(actions)
        self.performance_metrics['cache_hits'].append(cache_hits)
        self.performance_metrics['unique_states'].append(len(unique_states))
        
        return T_new, Y_new, actions

    def solve(self):
        """Solve until end time"""
        t = 0.0
        n_steps = int(self.end_time / self.dt)
        
        # Storage for results
        times = [t]
        temperatures = [self.T.copy()]
        species = [self.Y.copy()]
        actions_history = []
        
        print(f"Starting simulation with {n_steps} time steps...")
        
        for step in range(n_steps):
            if step % 10 == 0:
                print(f"Step {step}/{n_steps}, t = {t:.3e}")
            
            T, Y, actions = self.solve_step(t)
            t += self.dt
            
            times.append(t)
            temperatures.append(T.copy())
            species.append(Y.copy())
            actions_history.append(actions)
            
        results = {
            'times': np.array(times),
            'temperatures': np.array(temperatures),
            'species': np.array(species),
            'actions': np.array(actions_history),
            'metrics': self.performance_metrics
        }
        
        return results

    def visualize_results(self, results):
        """Visualize simulation results"""
        times = results['times']
        temperatures = results['temperatures']
        actions = results['actions']
        
        if self.n_dim == 1:
            self._plot_1d_results(times, temperatures, actions)
        elif self.n_dim == 2:
            self._plot_2d_results(times, temperatures, actions)
        else:
            print("Visualization for 3D results not implemented")

    def _plot_1d_results(self, times, temperatures, actions):
        """Plot results for 1D simulation"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
        
        # Temperature evolution
        x_mesh, t_mesh = np.meshgrid(self.grids[0], times)
        temp = ax1.pcolormesh(x_mesh, t_mesh, temperatures, shading='auto', cmap='hot')
        fig.colorbar(temp, ax=ax1, label='Temperature (K)')
        ax1.set_xlabel('Position (m)')
        ax1.set_ylabel('Time (s)')
        ax1.set_title('Temperature Evolution')
        
        # Integrator selection
        im = ax2.pcolormesh(x_mesh, t_mesh, actions, shading='auto')
        fig.colorbar(im, ax=ax2, label='Integrator Index')
        ax2.set_xlabel('Position (m)')
        ax2.set_ylabel('Time (s)')
        ax2.set_title('Integrator Selection')
        
        # Performance metrics
        cache_hits = self.performance_metrics['cache_hits']
        parallel_overhead = self.performance_metrics['parallel_overhead']
        
        ax3.plot(times[1:], cache_hits, label='Cache Hits')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Count')
        ax3.set_title('Performance Metrics')
        ax3.legend()
        
        plt.tight_layout()
        return fig

    def _plot_2d_results(self, times, temperatures, actions):
        """Plot results for 2D simulation"""
        # Select time points for visualization
        time_indices = [0, len(times)//2, -1]
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        for i, t_idx in enumerate(time_indices):
            # Temperature plot
            temp = axes[0,i].pcolormesh(self.grids[0], self.grids[1], 
                                      temperatures[t_idx], cmap='hot')
            fig.colorbar(temp, ax=axes[0,i], label='Temperature (K)')
            axes[0,i].set_title(f't = {times[t_idx]:.2e} s')
            
            # Integrator selection plot
            im = axes[1,i].pcolormesh(self.grids[0], self.grids[1], 
                                    actions[t_idx], cmap='viridis')
            fig.colorbar(im, ax=axes[1,i], label='Integrator Index')
            
        plt.tight_layout()
        return fig

# Example usage:
if __name__ == "__main__":
    # Load your trained policy
    policy_model = torch.load('your_policy.pt')
    
    # Create solver
    solver = PolicyGuidedSolverND(
        policy_model=policy_model,
        mech_file='mechanism_files/ch4_53species.yaml',
        initial_state={'T': 1400, 'P': ct.one_atm*40, 
                      'X': 'CH4:1, O2:2, N2:7.52'},
        dimensions=(100,),  # 1D example
        domain_size=(0.02,),
        end_time=2e-4,
        dt=1e-6,
        species_to_track=['H2', 'O2', 'H', 'OH', 'O2', 'H2O', 'HO2', 'N2', 'H2O2'],
        features_config={
            'temporal_features': True,
            'species_features': True,
            'stiffness_features': False,
            'basic_features': True,
            'include_dt_etol': False,
            'add_net_production_rates': False
        }
    )
    
    # Run simulation
    results = solver.solve()
    
    # Visualize results
    solver.visualize_results(results)
    plt.show()
                                        