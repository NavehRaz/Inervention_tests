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

### Trajectory Saving
- Save damage trajectories for each simulated individual
- Configurable number of time points or specific time points
- Support for equally spaced points or custom time arrays

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

## Requirements

- NumPy
- Numba
- Joblib (for parallel processing)
- SRtools (custom module for SR models)

## Installation

1. Clone the repository:
```bash
git clone git@github.com:NavehRaz/Inervention_tests.git
cd Inervention_tests
```

2. Install required dependencies:
```bash
pip install numpy numba joblib
```

3. Ensure SRtools is available in your Python path.

## Contributing

This repository is part of ongoing research in aging interventions. For questions or contributions, please open an issue or contact the repository maintainer.

## License

This project is part of academic research. Please cite appropriately if used in publications.
