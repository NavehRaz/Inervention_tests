# Intervention Tests

A comprehensive simulation framework for testing various interventions in aging models using stochastic differential equations.

## Overview

This repository contains tools for simulating and analyzing different types of interventions in aging processes. The main component is `intervention_SR.py`, which implements a flexible intervention system with support for trajectory saving and multiple intervention types.

## Features

### Intervention Types
- **Transient**: Instantaneous damage reduction at intervention time
- **Damage removal**: Continuous damage removal during intervention duration
- **Beta change**: Modifies beta parameter during intervention
- **Eta change**: Modifies eta parameter during intervention
- **Xc change**: Modifies xc parameter during intervention
- **Epsilon change**: Modifies epsilon parameter during intervention
- **Alpha**: Adds -alpha*x term to the equation during intervention
- **Saturating_eta**: Modifies eta by intervention_effect*(eta-eta_min) with time unit support

### Trajectory Saving & Plotting
- Save damage trajectories for each simulated individual
- Configurable number of time points or specific time points
- Support for equally spaced points or custom time arrays
- Built-in plotting functionality with customizable options:
  - Choose number of trajectories to display
  - Mark death times with markers
  - Random or sequential trajectory selection
  - Customizable colormaps and styling
  - Trajectories stop at death time (no post-death plotting)
  - Show damage level at death as Xc (default behavior)
  - Clean plot appearance (removed top/right spines)

### Computational Methods
- Euler method with standard integration
- Brownian bridge crossing detection for improved accuracy
- Parallel and non-parallel execution options
- JIT compilation with Numba for performance

## Files

- `intervention_SR.py`: Main simulation engine with intervention system
- `Life_long_interventions.ipynb`: Analysis of long-term interventions
- `Transient_intervension.ipynb`: Analysis of transient interventions
- `drosophila.ipynb`: Drosophila-specific analysis
- `trajectory_plotting_example.py`: Example script demonstrating plotting functionality

## Usage

### Basic Usage

```python
from intervention_SR import getInterventionSR

# Define model parameters
theta = [eta, beta, epsilon, xc]

# Create simulation with intervention
sim = getInterventionSR(
    theta,
    intervention_type='Alpha',
    intervention_effect=-0.1,
    intervention_duration=[10, 20],  # From time 10 to 20
    save_trajectory=True,
    traj_points=500  # Save 500 equally spaced points
)

# Run simulation
death_times, events = sim.calc_death_times()

# Access trajectories
trajectories = sim.trajectories  # Shape: (npeople, ntraj_points)
time_points = sim.traj_time_points

# Plot trajectories
ax = sim.plot_trajectories(n_trajectories=10, mark_death=True, colormap='viridis')
```

### Saturating_eta with Time Units

```python
# Using days (default)
sim = getInterventionSR(theta, intervention_type='Saturating_eta', intervention_effect=-0.05)

# Using years
sim = getInterventionSR(theta, intervention_type='Saturating_eta', intervention_effect=[-0.05, 'years'])

# Using generations
sim = getInterventionSR(theta, intervention_type='Saturating_eta', intervention_effect=[-0.05, 'generations'])
```

### Custom Trajectory Points

```python
# Save at specific time points
sim = getInterventionSR(
    theta,
    save_trajectory=True,
    traj_points=[1, 5, 10, 15, 25, 50]  # Specific times
)
```

### Plotting Options

```python
# Basic plotting
ax = sim.plot_trajectories()

# Customize plotting options
ax = sim.plot_trajectories(
    n_trajectories=15,           # Number of trajectories to plot
    mark_death=True,             # Mark death times with 'x'
    random_selection=False,      # Use first n trajectories instead of random
    colormap='plasma',           # Matplotlib colormap
    alpha=0.7,                   # Line transparency
    linewidth=1.5,               # Line width
    show_death_level=True        # Show damage at death as Xc level (default: True)
)

# Use existing axes
fig, ax = plt.subplots(figsize=(12, 8))
sim.plot_trajectories(ax=ax, n_trajectories=20, colormap='tab20')
ax.set_title('My Custom Title')
```

## Requirements

- NumPy
- Numba
- Joblib (for parallel processing)
- Matplotlib (for plotting)
- SRtools (custom module for SR models)

## Installation

1. Clone the repository:
```bash
git clone git@github.com:NavehRaz/Inervention_tests.git
cd Inervention_tests
```

2. Install required dependencies:
```bash
pip install numpy numba joblib matplotlib
```

3. Ensure SRtools is available in your Python path.

## Contributing

This repository is part of ongoing research in aging interventions. For questions or contributions, please open an issue or contact the repository maintainer.

## License

This project is part of academic research. Please cite appropriately if used in publications.
