from SRtools import SR_hetro as srh
import numpy as np
from numba import jit
from joblib import Parallel, delayed
from SRtools import deathTimesDataSet as dtds
import os
from SRtools import sr_mcmc as srmc
from SRtools import SRmodellib as sr
import matplotlib.pyplot as plt
import matplotlib.cm as cm

jit_nopython = True
eta_min = 2e-5  # Global variable for Saturating_eta intervention

def get_eta_min_for_time_unit(time_unit='days'):
    """
    Get the appropriate eta_min value based on time unit for Saturating_eta intervention.
    
    Parameters:
        time_unit (str): Time unit ('days', 'years', or 'generations')
    
    Returns:
        float: Scaled eta_min value
    """
    if time_unit == 'days':
        return eta_min
    elif time_unit == 'years':
        return eta_min * (365**2)
    elif time_unit == 'generations':
        return eta_min * (3/24)
    else:
        # Default to days if unknown time unit
        return eta_min



class intervention_SR(srh.SR_Hetro):
    def __init__(self, eta, beta, kappa, epsilon, xc, npeople, nsteps, t_end, 
                 eta_var = 0, beta_var = 0, kappa_var =0, epsilon_var =0, xc_var =0, t_start=0, tscale='years', external_hazard=np.inf, time_step_multiplier=1, parallel=False, bandwidth=3, method='brownian_bridge',
                 intervention_time=0, intervention_duration=0,intervention_type='Transient', intervention_effect=0, save_trajectory=False, traj_points=500):
        
        self.intervention_time = intervention_time
        self.intervention_duration = intervention_duration
        self.intervention_effect = intervention_effect
        self.intervention_type = intervention_type
        self.save_trajectory = save_trajectory
        self.traj_points = traj_points
        
        super().__init__(eta, beta, kappa, epsilon, xc, npeople, nsteps, t_end, eta_var, beta_var, kappa_var, epsilon_var, xc_var, t_start, tscale, external_hazard, time_step_multiplier, parallel, bandwidth, method=method)



    def _process_intervention_duration(self):
        """
        Process intervention_duration to extract start and stop times for JIT compatibility.
        Returns arrays of start and stop times for multiple intervals.
        """
        if isinstance(self.intervention_duration, (int, float)):
            # Single duration - not used for range-based interventions
            return np.array([0.0]), np.array([0.0])
        elif hasattr(self.intervention_duration, 'ndim'):
            if self.intervention_duration.ndim == 1 and len(self.intervention_duration) == 2:
                # Single interval: [start, stop]
                return np.array([float(self.intervention_duration[0])]), np.array([float(self.intervention_duration[1])])
            elif self.intervention_duration.ndim == 2 and self.intervention_duration.shape[0] > 0:
                # Multiple intervals: [[start_1,stop_1], [start_2,stop_2], ...]
                starts = np.array([float(self.intervention_duration[i, 0]) for i in range(self.intervention_duration.shape[0])])
                stops = np.array([float(self.intervention_duration[i, 1]) for i in range(self.intervention_duration.shape[0])])
                return starts, stops
        elif isinstance(self.intervention_duration, (list, tuple)):
            if len(self.intervention_duration) == 2 and not isinstance(self.intervention_duration[0], (list, tuple)):
                # Single interval as list/tuple: [start, stop]
                return np.array([float(self.intervention_duration[0])]), np.array([float(self.intervention_duration[1])])
            else:
                # Multiple intervals as list of lists: [[start_1,stop_1], [start_2,stop_2], ...]
                starts = np.array([float(interval[0]) for interval in self.intervention_duration])
                stops = np.array([float(interval[1]) for interval in self.intervention_duration])
                return starts, stops
        
        return np.array([0.0]), np.array([0.0])

    def _process_traj_points(self):
        """
        Process traj_points to extract time points for trajectory saving.
        Returns array of time points for JIT compatibility.
        """
        if not self.save_trajectory:
            return np.array([0.0])
        
        if isinstance(self.traj_points, (int, float)):
            # Number of equally spaced points
            num_points = int(self.traj_points)
            return np.linspace(0, self.t_end, num_points)
        elif hasattr(self.traj_points, '__iter__'):
            # Specific time points
            return np.array([float(t) for t in self.traj_points])
        else:
            # Default case
            return np.linspace(0, self.t_end, 500)

    def calc_death_times(self):
        s = len(self.t)
        dt = self.t[1]-self.t[0]
        sdt = np.sqrt(dt)
        t = self.t
        
        # Convert intervention type to numeric flag
        intervention_type_flag = 0
        if self.intervention_type == 'Transient':
            intervention_type_flag = 1
        elif self.intervention_type == 'Damage removal':
            intervention_type_flag = 2
        elif self.intervention_type == 'Beta change':
            intervention_type_flag = 3
        elif self.intervention_type == 'Eta change':
            intervention_type_flag = 4
        elif self.intervention_type == 'Xc change':
            intervention_type_flag = 5
        elif self.intervention_type == 'Epsilon change':
            intervention_type_flag = 6
        elif self.intervention_type == 'Alpha':
            intervention_type_flag = 7
        elif self.intervention_type == 'Saturating_eta':
            intervention_type_flag = 8
        
        # Parse intervention_effect for Saturating_eta to extract time unit and calculate eta_min
        saturating_eta_min = eta_min
        intervention_effect_value = self.intervention_effect
        if self.intervention_type == 'Saturating_eta':
            if isinstance(self.intervention_effect, (list, tuple)) and len(self.intervention_effect) == 2:
                intervention_effect_value = self.intervention_effect[0]
                time_unit = self.intervention_effect[1]
                saturating_eta_min = get_eta_min_for_time_unit(time_unit)
            # For other interventions, use intervention_effect as is
        
        # Process intervention duration for JIT compatibility
        intervention_duration_starts, intervention_duration_stops = self._process_intervention_duration()
        
        # Process trajectory points for saving
        traj_time_points = self._process_traj_points()
        
        if self.method == 'brownian_bridge':
            if self.parallel:
                death_times, events, trajectories = death_times_euler_brownian_bridge_parallel(s, dt, t, self.eta, self.eta_var, self.beta, self.beta_var, self.kappa, self.kappa_var, self.epsilon, self.epsilon_var, self.xc, self.xc_var, sdt, self.npeople, self.external_hazard, self.time_step_multiplier, self.intervention_time, intervention_type_flag, intervention_duration_starts, intervention_duration_stops, intervention_effect_value, intervention_effect_value, saturating_eta_min, self.save_trajectory, traj_time_points)
            else:
                death_times, events, trajectories = death_times_euler_brownian_bridge(s, dt, t, self.eta, self.eta_var, self.beta, self.beta_var, self.kappa, self.kappa_var, self.epsilon, self.epsilon_var, self.xc, self.xc_var, sdt, self.npeople, self.external_hazard, self.time_step_multiplier, self.intervention_time, intervention_type_flag, intervention_duration_starts, intervention_duration_stops, intervention_effect_value, intervention_effect_value, saturating_eta_min, self.save_trajectory, traj_time_points)
        elif self.method == 'euler':
            if self.parallel:
                death_times, events, trajectories = death_times_accelerator2(s, dt, t, self.eta, self.eta_var, self.beta, self.beta_var, self.kappa, self.kappa_var, self.epsilon, self.epsilon_var, self.xc, self.xc_var, sdt, self.npeople, self.external_hazard, self.time_step_multiplier, self.intervention_time, intervention_type_flag, self.intervention_duration, intervention_effect_value, intervention_effect_value, saturating_eta_min, self.save_trajectory, traj_time_points)
            else:
                death_times, events, trajectories = death_times_accelerator(s, dt, t, self.eta, self.eta_var, self.beta, self.beta_var, self.kappa, self.kappa_var, self.epsilon, self.epsilon_var, self.xc, self.xc_var, sdt, self.npeople, self.external_hazard, self.time_step_multiplier, self.intervention_time, intervention_type_flag, intervention_duration_starts, intervention_duration_stops, intervention_effect_value, intervention_effect_value, saturating_eta_min, self.save_trajectory, traj_time_points)
        else:
            # Default to brownian bridge if method not recognized
            if self.parallel:
                death_times, events, trajectories = death_times_euler_brownian_bridge_parallel(s, dt, t, self.eta, self.eta_var, self.beta, self.beta_var, self.kappa, self.kappa_var, self.epsilon, self.epsilon_var, self.xc, self.xc_var, sdt, self.npeople, self.external_hazard, self.time_step_multiplier, self.intervention_time, intervention_type_flag, intervention_duration_starts, intervention_duration_stops, intervention_effect_value, intervention_effect_value, saturating_eta_min, self.save_trajectory, traj_time_points)
            else:
                death_times, events, trajectories = death_times_euler_brownian_bridge(s, dt, t, self.eta, self.eta_var, self.beta, self.beta_var, self.kappa, self.kappa_var, self.epsilon, self.epsilon_var, self.xc, self.xc_var, sdt, self.npeople, self.external_hazard, self.time_step_multiplier, self.intervention_time, intervention_type_flag, intervention_duration_starts, intervention_duration_stops, intervention_effect_value, intervention_effect_value, saturating_eta_min, self.save_trajectory, traj_time_points)

        # Store results
        self.death_times = np.array(death_times)
        self.events = np.array(events)
        
        if self.save_trajectory:
            self.trajectories = trajectories
            self.traj_time_points = traj_time_points
        
        return self.death_times, self.events
    
    def plot_trajectories(self, n_trajectories=10, mark_death=True, random_selection=True, 
                         colormap='viridis', ax=None, alpha=0.7, linewidth=1.0, show_death_level=True):
        """
        Plot damage trajectories from the simulation.
        
        Parameters:
            n_trajectories (int): Number of trajectories to plot (default: 10)
            mark_death (bool): Whether to mark death times with markers (default: True)
            random_selection (bool): Whether to select trajectories randomly or first n (default: True)
            colormap (str): Matplotlib colormap name for trajectory colors (default: 'viridis')
            ax (matplotlib.axes.Axes): Axes object to plot on. If None, creates new figure (default: None)
            alpha (float): Transparency of trajectory lines (default: 0.7)
            linewidth (float): Width of trajectory lines (default: 1.0)
            show_death_level (bool): Whether to show damage level at death as Xc (default: True)
        
        Returns:
            matplotlib.axes.Axes: The axes object with the plot
            
        Raises:
            ValueError: If trajectories haven't been saved or if n_trajectories is invalid
        """
        if not hasattr(self, 'trajectories') or self.trajectories is None:
            raise ValueError("No trajectories available. Set save_trajectory=True when creating the simulation.")
        
        if not hasattr(self, 'traj_time_points') or self.traj_time_points is None:
            raise ValueError("No trajectory time points available.")
        
        n_people = self.trajectories.shape[0]
        if n_trajectories > n_people:
            n_trajectories = n_people
            print(f"Warning: Requested {n_trajectories} trajectories but only {n_people} available. Plotting all {n_people}.")
        
        if n_trajectories <= 0:
            raise ValueError("n_trajectories must be positive.")
        
        # Create figure and axes if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        # Select trajectories to plot
        if random_selection:
            selected_indices = np.random.choice(n_people, size=n_trajectories, replace=False)
        else:
            selected_indices = np.arange(n_trajectories)
        
        # Get colormap
        cmap = cm.get_cmap(colormap)
        colors = cmap(np.linspace(0, 1, n_trajectories))
        
        # Plot trajectories
        for i, idx in enumerate(selected_indices):
            trajectory = self.trajectories[idx, :].copy()
            time_points = self.traj_time_points.copy()
            
            # Check if this individual died and truncate trajectory at death
            if hasattr(self, 'death_times') and hasattr(self, 'events'):
                if idx < len(self.events) and self.events[idx] == 1:  # Death event occurred
                    death_time = self.death_times[idx]
                    
                    # Find trajectory points before or at death time
                    valid_indices = time_points <= death_time
                    
                    if np.any(valid_indices):
                        # Use only trajectory points up to death
                        time_points = time_points[valid_indices]
                        trajectory = trajectory[valid_indices]
                        
                        # Add death point at exact death time with Xc level if show_death_level is True
                        if show_death_level:
                            time_points = np.append(time_points, death_time)
                            trajectory = np.append(trajectory, self.xc)
            
            # Plot the trajectory
            ax.plot(time_points, trajectory, color=colors[i], 
                   alpha=alpha, linewidth=linewidth, label=f'Individual {idx}' if n_trajectories <= 10 else None)
            
            # Mark death time if requested
            if mark_death and hasattr(self, 'death_times') and hasattr(self, 'events'):
                if idx < len(self.events) and self.events[idx] == 1:  # Death event occurred
                    death_time = self.death_times[idx]
                    death_damage = self.xc if show_death_level else trajectory[-1]
                    ax.scatter(death_time, death_damage, color=colors[i], s=50, 
                             marker='x', alpha=1.0, zorder=5)
        
        # Customize plot appearance
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('Time')
        ax.set_ylabel('Damage')
        ax.set_title(f'Damage Trajectories (n={n_trajectories})')
        ax.grid(True, alpha=0.3)
        
        # Add legend if not too many trajectories
        if n_trajectories <= 10:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add death markers to legend if applicable
        if mark_death:
            death_label = 'Death time (Xc level)' if show_death_level else 'Death time'
            ax.scatter([], [], color='black', marker='x', s=50, alpha=1.0, label=death_label)
            if n_trajectories <= 10:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        return ax
    

def getInterventionSR(
    theta,
    n=25000,
    nsteps=6000,
    t_end=110,
    external_hazard=np.inf,
    time_step_multiplier=1,
    npeople=None,
    parallel=False,
    eta_var=0,
    beta_var=0,
    epsilon_var=0,
    xc_var=0.2,
    kappa_var=0,
    hetro=False,
    bandwidth=3,
    step_size=None,
    method='brownian_bridge',
    intervention_time=0,
    intervention_duration=0,
    intervention_type='Transient',
    intervention_effect=0,
    save_trajectory=False,
    traj_points=500
    ):
    """
    Optionally specify step_size. If step_size is given, nsteps and time_step_multiplier are ignored and recalculated so that
    t_end/(nsteps*time_step_multiplier) = step_size. If nsteps*time_step_multiplier <= 6000, time_step_multiplier=1, else
    increase time_step_multiplier until nsteps <= 6000. Both nsteps and time_step_multiplier are integers.

    Parameters:
        method (str): Method to use for death times calculation. Options:
            - 'brownian_bridge': Euler method with Brownian bridge crossing detection (default)
            - 'euler': Standard Euler method
        intervention_time (float): Time at which intervention occurs.
        intervention_duration (float, list, or array): Duration of intervention. Can be:
            - Single number (duration from intervention_time for 'Transient' type)
            - [start, stop] for single interval interventions
            - [[start_1,stop_1], [start_2,stop_2], ...] for multiple intervals
        intervention_type (str): Type of intervention. Options:
            - 'Transient': Instantaneous damage reduction at intervention_time
            - 'Damage removal': Continuous damage removal during duration
            - 'Beta change': Modifies beta parameter during duration
            - 'Eta change': Modifies eta parameter during duration
            - 'Xc change': Modifies xc parameter during duration
            - 'Epsilon change': Modifies epsilon parameter during duration
            - 'Alpha': Adds -alpha*x term to the equation during duration
            - 'Saturating_eta': Modifies eta by intervention_effect*(eta-eta_min) during duration
        intervention_effect (float or list): Effect size of intervention. For Saturating_eta, can be:
            - float: saturation factor (assumes 'days' time unit)
            - [value, 'days']: saturation factor with days time unit (uses eta_min)
            - [value, 'years']: saturation factor with years time unit (uses eta_min*(365^2))
            - [value, 'generations']: saturation factor with generations time unit (uses eta_min*(3/24))
            For other interventions: percentage change, alpha value for Alpha intervention.
        save_trajectory (bool): Whether to save damage trajectories for each individual (default: False).
        traj_points (int or array-like): Number of trajectory points to save or specific time points.
            - int: Number of equally spaced points between 0 and t_end (default: 500)
            - array-like: Specific time points to save (e.g., [1, 10, 12.5])
    
    Returns:
        intervention_SR: Simulation object with methods:
            - calc_death_times(): Run simulation and return death times and events
            - plot_trajectories(): Plot damage trajectories (requires save_trajectory=True)
    """
    if npeople is not None:
        n = npeople
    eta = theta[0]
    beta = theta[1]
    epsilon = theta[2]
    xc = theta[3]
    if not hetro:
        eta_var = 0.0
        beta_var = 0.0
        epsilon_var = 0.0
        xc_var = 0.0
        kappa_var = 0.0
    else:
        # Ensure all variance parameters are floats
        eta_var = float(eta_var)
        beta_var = float(beta_var)
        epsilon_var = float(epsilon_var)
        xc_var = float(xc_var)
        kappa_var = float(kappa_var)

    if external_hazard is None or external_hazard == 'None':
        external_hazard = np.inf

    # Handle step_size logic
    if step_size is not None:
        total_steps = int(np.ceil(t_end / step_size))
        if total_steps <= 6000:
            nsteps = total_steps
            time_step_multiplier = 1
        else:
            # Find smallest integer time_step_multiplier so that nsteps <= 6000
            time_step_multiplier = int(np.ceil(total_steps / 6000))
            nsteps = int(np.ceil(total_steps / time_step_multiplier))
            # Ensure both are at least 1
            time_step_multiplier = max(1, time_step_multiplier)
            nsteps = max(1, nsteps)
    if isinstance(intervention_effect, (list, tuple)) and len(intervention_effect) == 2:
        intervention_effect = [float(intervention_effect[0]),intervention_effect[1]]
    else:
        intervention_effect= float(intervention_effect)

    sim = intervention_SR(
        eta=eta,
        beta=beta,
        epsilon=epsilon,
        xc=xc,
        eta_var=eta_var,
        beta_var=beta_var,
        kappa_var=kappa_var,
        epsilon_var=epsilon_var,
        xc_var=xc_var,
        kappa=0.5,
        npeople=n,
        nsteps=nsteps,
        t_end=t_end,
        external_hazard=external_hazard,
        time_step_multiplier=time_step_multiplier,
        parallel=parallel,
        bandwidth=bandwidth,
        method=method,
        intervention_time=float(intervention_time),
        intervention_duration=intervention_duration,
        intervention_type=intervention_type,
        intervention_effect=intervention_effect,
        save_trajectory=save_trajectory,
        traj_points=traj_points
    )

    return sim



@jit(nopython=jit_nopython)
def is_within_intervention_duration(current_time, intervention_duration):
    """
    Check if current_time is within any intervention duration.
    intervention_duration can be:
    - A single number (duration from intervention_time) - returns False for range-based interventions
    - A 1D array of two numbers [start, stop] 
    - A 2D array of intervals [[start_1,stop_1], [start_2,stop_2], ...]
    """
    # Check if intervention_duration is a scalar (single number)
    if intervention_duration.ndim == 0:
        # Single duration - not used for range-based interventions
        return False
    
    # Handle 2D array of intervals: [[start_1,stop_1], [start_2,stop_2], ...]
    if intervention_duration.ndim == 2:
        for i in range(intervention_duration.shape[0]):
            start_time = intervention_duration[i, 0]
            stop_time = intervention_duration[i, 1]
            if start_time <= current_time <= stop_time:
                return True
    # Handle 1D array: [start, stop]
    elif intervention_duration.ndim == 1 and intervention_duration.shape[0] == 2:
        start_time = intervention_duration[0]
        stop_time = intervention_duration[1]
        if start_time <= current_time <= stop_time:
            return True
    
    return False

@jit(nopython=jit_nopython)
def check_intervention_duration_simple(current_time, start_time, stop_time):
    """
    Simple check for single interval intervention duration.
    """
    return start_time <= current_time <= stop_time

@jit(nopython=jit_nopython)
def is_within_any_intervention_interval(current_time, intervention_starts, intervention_stops):
    """
    Check if current_time is within any of the intervention intervals.
    intervention_starts and intervention_stops are arrays of start and stop times.
    """
    for i in range(len(intervention_starts)):
        if intervention_starts[i] <= current_time <= intervention_stops[i]:
            return True
    return False

#method without parallelization (for cluster usage)
@jit(nopython=jit_nopython)
def death_times_accelerator(s,dt,t,eta0,eta_var,beta0,beta_var,kappa0,kappa_var,epsilon0,epsilon_var,xc0,xc_var,sdt,npeople,external_hazard = np.inf,time_step_multiplier = 1, intervention_time=0, intervention_type_flag=0, intervention_duration_starts=np.array([0.0]), intervention_duration_stops=np.array([0.0]), intervention_effect=0, saturating_eta_min=2e-5, save_trajectory=False, traj_time_points=np.array([0.0])):
    death_times = []
    events = []
    if save_trajectory:
        trajectories = np.zeros((npeople, len(traj_time_points)))
    else:
        trajectories = np.zeros((1, 1))  # Dummy array for JIT compatibility
    
    for l in range(npeople):
        x=0
        j=0
        ndt = dt/time_step_multiplier
        nsdt = sdt/np.sqrt(time_step_multiplier)
        chance_to_die_externally = np.exp(-external_hazard)*ndt
        eta = eta0*np.random.normal(1.0, eta_var)
        beta = beta0 * np.random.normal(1.0, beta_var)
        kappa = kappa0 * np.random.normal(1.0, kappa_var)
        epsilon = epsilon0 * np.random.normal(1.0, epsilon_var)
        xc = xc0 * np.random.normal(1.0, xc_var)
        while j in range(s-1) and x<xc:
            current_time = t[j]
            
            # Apply interventions based on type
            if intervention_type_flag == 1 and abs(current_time - intervention_time) < dt/2:
                # Transient intervention
                x = x * (1 - intervention_effect)
                x = np.maximum(x, 0)
            elif intervention_type_flag == 2 and is_within_any_intervention_interval(current_time, intervention_duration_starts, intervention_duration_stops):
                # Damage removal intervention
                x = x * (1 - intervention_effect)
                x = np.maximum(x, 0)
            
            # Save trajectory if requested (after interventions, before simulation steps)
            if save_trajectory and x < xc:  # Only save if still alive
                for k in range(len(traj_time_points)):
                    if abs(current_time - traj_time_points[k]) < dt/2:
                        trajectories[l, k] = x
            
            for i in range(time_step_multiplier):
                # Apply parameter modifications during intervention
                current_eta = eta
                current_beta = beta
                current_epsilon = epsilon
                current_xc = xc
                if intervention_type_flag == 3 and is_within_any_intervention_interval(current_time, intervention_duration_starts, intervention_duration_stops):
                    # Beta change intervention
                    current_beta = beta * (1 + intervention_effect)
                elif intervention_type_flag == 4 and is_within_any_intervention_interval(current_time, intervention_duration_starts, intervention_duration_stops):
                    # Eta change intervention
                    current_eta = eta * (1 + intervention_effect)
                elif intervention_type_flag == 5 and is_within_any_intervention_interval(current_time, intervention_duration_starts, intervention_duration_stops):
                    # Xc change intervention
                    current_xc = xc * (1 + intervention_effect)
                elif intervention_type_flag == 6 and is_within_any_intervention_interval(current_time, intervention_duration_starts, intervention_duration_stops):
                    # Epsilon change intervention
                    current_epsilon = epsilon * (1 + intervention_effect)
                elif intervention_type_flag == 8 and is_within_any_intervention_interval(current_time, intervention_duration_starts, intervention_duration_stops):
                    # Saturating_eta intervention
                    current_eta = eta + intervention_effect * (eta - saturating_eta_min)
                
                # Apply Alpha intervention (add -alpha*x term)
                alpha_term = 0.0
                if intervention_type_flag == 7 and is_within_any_intervention_interval(current_time, intervention_duration_starts, intervention_duration_stops):
                    alpha_term = -intervention_effect * x
                
                noise = np.sqrt(2*current_epsilon)*np.random.normal(0.0, 1.0)
                x = x+ndt*(current_eta*(t[j]+i*ndt)-current_beta*x/(x+kappa)+alpha_term)+noise*nsdt
                x = np.maximum(x, 0)
                if np.random.uniform(0,1)<chance_to_die_externally:
                    x = current_xc
                if x>=current_xc:
                    break
            j+=1
        if x>=xc:
            death_times.append(j*dt)
            events.append(1)
        else:
            death_times.append(j*dt)
            events.append(0)

    return death_times, events, trajectories

##method with parallelization (run on your computer)
def death_times_accelerator2(s,dt,t,eta,eta_var,beta,beta_var,kappa,kappa_var,epsilon,epsilon_var,xc,xc_var,sdt,npeople,external_hazard = np.inf,time_step_multiplier = 1, intervention_time=0, intervention_type_flag=0, intervention_duration=0, intervention_effect=0, saturating_eta_min=2e-5, save_trajectory=False, traj_time_points=np.array([0.0])):
    # Process intervention duration for JIT compatibility
    if isinstance(intervention_duration, (int, float)):
        intervention_duration_starts, intervention_duration_stops = np.array([0.0]), np.array([0.0])
    elif hasattr(intervention_duration, 'ndim'):
        if intervention_duration.ndim == 1 and len(intervention_duration) == 2:
            intervention_duration_starts, intervention_duration_stops = np.array([float(intervention_duration[0])]), np.array([float(intervention_duration[1])])
        elif intervention_duration.ndim == 2 and intervention_duration.shape[0] > 0:
            intervention_duration_starts = np.array([float(intervention_duration[i, 0]) for i in range(intervention_duration.shape[0])])
            intervention_duration_stops = np.array([float(intervention_duration[i, 1]) for i in range(intervention_duration.shape[0])])
        else:
            intervention_duration_starts, intervention_duration_stops = np.array([0.0]), np.array([0.0])
    elif isinstance(intervention_duration, (list, tuple)):
        if len(intervention_duration) == 2 and not isinstance(intervention_duration[0], (list, tuple)):
            intervention_duration_starts, intervention_duration_stops = np.array([float(intervention_duration[0])]), np.array([float(intervention_duration[1])])
        else:
            intervention_duration_starts = np.array([float(interval[0]) for interval in intervention_duration])
            intervention_duration_stops = np.array([float(interval[1]) for interval in intervention_duration])
    else:
        intervention_duration_starts, intervention_duration_stops = np.array([0.0]), np.array([0.0])
    
    @jit(nopython=jit_nopython)
    def calculate_death_times(npeople, s, dt, t, eta0,eta_var,beta0,beta_var,kappa0,kappa_var,epsilon0,epsilon_var,xc0,xc_var, sdt, external_hazard,time_step_multiplier, intervention_time, intervention_type_flag, intervention_duration_starts, intervention_duration_stops, intervention_effect, saturating_eta_min, save_trajectory, traj_time_points):
        death_times = []
        events =[]
        if save_trajectory:
            trajectories = np.zeros((npeople, len(traj_time_points)))
        else:
            trajectories = np.zeros((1, 1))  # Dummy array for JIT compatibility
        for l in range(npeople):
            died = False
            x = 0
            j = 0
            ndt = dt/time_step_multiplier
            nsdt = np.sqrt(ndt)
            chance_to_die_externally = np.exp(-external_hazard)*ndt
            eta = eta0*np.random.normal(1.0, eta_var)
            beta = beta0 * np.random.normal(1.0, beta_var)
            kappa = kappa0 * np.random.normal(1.0, kappa_var)
            epsilon = epsilon0 * np.random.normal(1.0, epsilon_var)
            xc = xc0 * np.random.normal(1.0, xc_var)
            while j in range(s - 1) and x < xc and not died:
                current_time = t[j]
                
                # Apply interventions based on type
                if intervention_type_flag == 1 and abs(current_time - intervention_time) < dt/2:
                    # Transient intervention
                    x = x * (1 - intervention_effect)
                    x = np.maximum(x, 0)
                elif intervention_type_flag == 2 and is_within_any_intervention_interval(current_time, intervention_duration_starts, intervention_duration_stops):
                    # Damage removal intervention
                    x = x * (1 - intervention_effect)
                    x = np.maximum(x, 0)
                
                # Save trajectory if requested (after interventions, before simulation steps)
                if save_trajectory and x < xc:  # Only save if still alive
                    for k in range(len(traj_time_points)):
                        if abs(current_time - traj_time_points[k]) < dt/2:
                            trajectories[l, k] = x
                
                for i in range(time_step_multiplier):
                    # Apply parameter modifications during intervention
                    current_eta = eta
                    current_beta = beta
                    current_epsilon = epsilon
                    current_xc = xc
                    if intervention_type_flag == 3 and is_within_any_intervention_interval(current_time, intervention_duration_starts, intervention_duration_stops):
                        # Beta change intervention
                        current_beta = beta * (1 + intervention_effect)
                    elif intervention_type_flag == 4 and is_within_any_intervention_interval(current_time, intervention_duration_starts, intervention_duration_stops):
                        # Eta change intervention
                        current_eta = eta * (1 + intervention_effect)
                    elif intervention_type_flag == 5 and is_within_any_intervention_interval(current_time, intervention_duration_starts, intervention_duration_stops):
                        # Xc change intervention
                        current_xc = xc * (1 + intervention_effect)
                    elif intervention_type_flag == 6 and is_within_any_intervention_interval(current_time, intervention_duration_starts, intervention_duration_stops):
                        # Epsilon change intervention
                        current_epsilon = epsilon * (1 + intervention_effect)
                    elif intervention_type_flag == 8 and is_within_any_intervention_interval(current_time, intervention_duration_starts, intervention_duration_stops):
                        # Saturating_eta intervention
                        current_eta = eta + intervention_effect * (eta - saturating_eta_min)
                    
                    # Apply Alpha intervention (add -alpha*x term)
                    alpha_term = 0.0
                    if intervention_type_flag == 7 and is_within_any_intervention_interval(current_time, intervention_duration_starts, intervention_duration_stops):
                        alpha_term = -intervention_effect * x
                    
                    noise = np.sqrt(2*current_epsilon)*np.random.normal(0.0, 1.0)
                    x = x+ndt*(current_eta*(t[j]+i*ndt)-current_beta*x/(x+kappa)+alpha_term)+noise*nsdt
                    x = np.maximum(x, 0)
                    if np.random.uniform(0,1)<chance_to_die_externally:
                        x = current_xc
                    if x>=current_xc:
                        died = True
                j += 1
            if died:
                death_times.append(j * dt)
                events.append(1)
            else:
                death_times.append(j * dt)
                events.append(0)
        return death_times, events, trajectories

    n_jobs = os.cpu_count()
    npeople_per_job = npeople // n_jobs
    results = Parallel(n_jobs=n_jobs)(delayed(calculate_death_times)(
        npeople_per_job, s, dt, t, eta,eta_var, beta,beta_var, kappa,kappa_var, epsilon,epsilon_var, xc,xc_var, sdt, external_hazard,time_step_multiplier, intervention_time, intervention_type_flag, intervention_duration_starts, intervention_duration_stops, intervention_effect, saturating_eta_min, save_trajectory, traj_time_points
    ) for _ in range(n_jobs))

    death_times = [dt for sublist in results for dt in sublist[0]]
    events = [event for sublist in results for event in sublist[1]]
    trajectories = np.vstack([sublist[2] for sublist in results])
    return death_times, events, trajectories



# Euler with Brownian Bridge method
@jit(nopython=jit_nopython)
def death_times_euler_brownian_bridge(s, dt, t, eta0, eta_var, beta0, beta_var, kappa0, kappa_var,
                                     epsilon0, epsilon_var, xc0, xc_var, sdt, npeople,
                                     external_hazard=np.inf, time_step_multiplier=1, intervention_time=0, intervention_type_flag=0, intervention_duration_starts=np.array([0.0]), intervention_duration_stops=np.array([0.0]), intervention_effect=0, saturating_eta_min=2e-5, save_trajectory=False, traj_time_points=np.array([0.0])):
    """
    Euler method with Brownian bridge crossing detection.
    This method uses the standard Euler scheme but adds Brownian bridge
    crossing probability tests to detect barrier crossings between time steps.
    """
    death_times = []
    events = []
    if save_trajectory:
        trajectories = np.zeros((npeople, len(traj_time_points)))
    else:
        trajectories = np.zeros((1, 1))  # Dummy array for JIT compatibility
    
    ndt = dt / time_step_multiplier
    nsdt = sdt / np.sqrt(time_step_multiplier)
    constant_hazard = np.isfinite(external_hazard)
    if constant_hazard:
        chance_to_die_externally = np.exp(-external_hazard) * ndt
    
    for person in range(npeople):
        x = 0.0
        j = 0
        eta = eta0 * np.random.normal(1.0, eta_var)
        beta = beta0 * np.random.normal(1.0, beta_var)
        kappa = kappa0 * np.random.normal(1.0, kappa_var)
        epsilon = epsilon0 * np.random.normal(1.0, epsilon_var)
        xc = xc0 * np.random.normal(1.0, xc_var)
        sqrt_2epsilon = np.sqrt(2 * epsilon)
        crossed = False
        
        while j < s - 1 and not crossed:
            current_time = t[j]
            
            # Apply interventions based on type
            if intervention_type_flag == 1 and abs(current_time - intervention_time) < dt/2:
                # Transient intervention
                x = x * (1 - intervention_effect)
                x = max(x, 0.0)
            elif intervention_type_flag == 2 and is_within_any_intervention_interval(current_time, intervention_duration_starts, intervention_duration_stops):
                # Damage removal intervention
                x = x * (1 - intervention_effect)
                x = max(x, 0.0)
            
            # Save trajectory if requested (after interventions, before simulation steps)
            if save_trajectory and x < xc:  # Only save if still alive
                for k in range(len(traj_time_points)):
                    if abs(current_time - traj_time_points[k]) < dt/2:
                        trajectories[person, k] = x
            
            for _ in range(time_step_multiplier):
                # Apply parameter modifications during intervention
                current_eta = eta
                current_beta = beta
                current_epsilon = epsilon
                current_xc = xc
                if intervention_type_flag == 3 and is_within_any_intervention_interval(current_time, intervention_duration_starts, intervention_duration_stops):
                    # Beta change intervention
                    current_beta = beta * (1 + intervention_effect)
                elif intervention_type_flag == 4 and is_within_any_intervention_interval(current_time, intervention_duration_starts, intervention_duration_stops):
                    # Eta change intervention
                    current_eta = eta * (1 + intervention_effect)
                elif intervention_type_flag == 5 and is_within_any_intervention_interval(current_time, intervention_duration_starts, intervention_duration_stops):
                    # Xc change intervention
                    current_xc = xc * (1 + intervention_effect)
                elif intervention_type_flag == 6 and is_within_any_intervention_interval(current_time, intervention_duration_starts, intervention_duration_stops):
                    # Epsilon change intervention
                    current_epsilon = epsilon * (1 + intervention_effect)
                elif intervention_type_flag == 8 and is_within_any_intervention_interval(current_time, intervention_duration_starts, intervention_duration_stops):
                    # Saturating_eta intervention
                    current_eta = eta + intervention_effect * (eta - saturating_eta_min)
                
                # Apply Alpha intervention (add -alpha*x term)
                alpha_term = 0.0
                if intervention_type_flag == 7 and is_within_any_intervention_interval(current_time, intervention_duration_starts, intervention_duration_stops):
                    alpha_term = -intervention_effect * x
                
                # Standard Euler step
                drift = current_eta * t[j] - current_beta * x / (x + kappa) + alpha_term
                noise = np.sqrt(2 * current_epsilon) * np.random.normal()
                x_new = x + ndt * drift + noise * nsdt
                x_new = max(x_new, 0.0)
                
                # Check external hazard
                if constant_hazard and np.random.rand() < chance_to_die_externally:
                    x = current_xc
                    crossed = True
                    break
                
                # Direct crossing check
                if x_new >= current_xc:
                    x = x_new
                    crossed = True
                    break
                
                # Brownian bridge crossing test if not crossed directly
                if (x < current_xc) and (x_new < current_xc) and (x > 0*kappa):
                    dx1 = current_xc - x
                    dx2 = current_xc - x_new
                    if dx1 > 0.0 and dx2 > 0.0:
                        # Brownian bridge crossing probability
                        # P = exp(-2 * (xc - x) * (xc - x_new) / (2 * epsilon * ndt))
                        var = 2.0 * current_epsilon * ndt
                        if var > 0.0:
                            p_cross = np.exp(-2.0 * dx1 * dx2 / var)
                            if np.random.rand() < p_cross:
                                x = current_xc
                                crossed = True
                                break
                
                x = x_new
            j += 1
        
        death_times.append(j * dt)
        if crossed or x >= current_xc:
            events.append(1)
        else:
            events.append(0)
    
    return np.array(death_times), np.array(events), trajectories

# Parallel version of Euler with Brownian Bridge method
def death_times_euler_brownian_bridge_parallel(s, dt, t, eta0, eta_var, beta0, beta_var, kappa0, kappa_var,
                                              epsilon0, epsilon_var, xc0, xc_var, sdt, npeople,
                                              external_hazard=np.inf, time_step_multiplier=1, intervention_time=0, intervention_type_flag=0, intervention_duration_starts=np.array([0.0]), intervention_duration_stops=np.array([0.0]), intervention_effect=0, intervention_effect_value=0, saturating_eta_min=2e-5, save_trajectory=False, traj_time_points=np.array([0.0]), n_jobs=-1, chunk_size=1000):
    """
    Parallel version of death_times_euler_brownian_bridge.
    Splits npeople into chunks and runs death_times_euler_brownian_bridge on each chunk in parallel.
    """
    from joblib import Parallel, delayed
    import numpy as np

    def worker(npeople_chunk, s, dt, t, eta0, eta_var, beta0, beta_var, kappa0, kappa_var,
               epsilon0, epsilon_var, xc0, xc_var, sdt, external_hazard, time_step_multiplier, intervention_time, intervention_type_flag, intervention_duration_starts, intervention_duration_stops, intervention_effect_value, saturating_eta_min, save_trajectory, traj_time_points):
        # Call the numba-jitted function for this chunk
        return death_times_euler_brownian_bridge(
            s, dt, t, eta0, eta_var, beta0, beta_var, kappa0, kappa_var,
            epsilon0, epsilon_var, xc0, xc_var, sdt, npeople_chunk,
            external_hazard, time_step_multiplier, intervention_time, intervention_type_flag, intervention_duration_starts, intervention_duration_stops, intervention_effect_value, saturating_eta_min, save_trajectory, traj_time_points
        )

    # Split npeople into chunks
    n_chunks = npeople // chunk_size
    remainder = npeople % chunk_size
    chunk_sizes = [chunk_size] * n_chunks
    if remainder > 0:
        chunk_sizes.append(remainder)

    results = Parallel(n_jobs=n_jobs)(
        delayed(worker)(
            n_chunk, s, dt, t, eta0, eta_var, beta0, beta_var, kappa0, kappa_var,
            epsilon0, epsilon_var, xc0, xc_var, sdt, external_hazard, time_step_multiplier, intervention_time, intervention_type_flag, intervention_duration_starts, intervention_duration_stops, intervention_effect_value, saturating_eta_min, save_trajectory, traj_time_points
        ) for n_chunk in chunk_sizes if n_chunk > 0
    )

    # Concatenate results
    death_times = np.concatenate([res[0] for res in results])
    events = np.concatenate([res[1] for res in results])
    trajectories = np.vstack([res[2] for res in results])
    return death_times, events, trajectories



