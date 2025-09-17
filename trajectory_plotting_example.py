"""
Example script demonstrating trajectory plotting functionality.
"""

import numpy as np
import matplotlib.pyplot as plt
from intervention_SR import getInterventionSR

# Example parameters
theta = [0.1, 0.5, 0.01, 1.0]  # [eta, beta, epsilon, xc]

def example_basic_plotting():
    """Basic trajectory plotting example."""
    print("Running basic trajectory plotting example...")
    
    # Create simulation with trajectory saving
    sim = getInterventionSR(
        theta,
        n=100,  # Small number for quick example
        t_end=50,
        save_trajectory=True,
        traj_points=100,  # 100 equally spaced points
        intervention_type='Alpha',
        intervention_effect=-0.1,
        intervention_duration=[10, 20]
    )
    
    # Run simulation
    death_times, events = sim.calc_death_times()
    
    # Plot trajectories with default settings
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Default plot (shows death level as Xc, stops at death)
    sim.plot_trajectories(ax=axes[0,0])
    axes[0,0].set_title('Default Settings (stops at death, shows Xc)')
    
    # More trajectories, no death markers
    sim.plot_trajectories(n_trajectories=20, mark_death=False, ax=axes[0,1])
    axes[0,1].set_title('20 trajectories, no death markers')
    
    # First 5 trajectories, no death level shown
    sim.plot_trajectories(n_trajectories=5, random_selection=False, 
                         colormap='plasma', show_death_level=False, ax=axes[1,0])
    axes[1,0].set_title('First 5 trajectories, no Xc level')
    
    # Custom styling with death level
    sim.plot_trajectories(n_trajectories=15, colormap='coolwarm', 
                         alpha=0.5, linewidth=2.0, show_death_level=True, ax=axes[1,1])
    axes[1,1].set_title('Custom styling (shows death at Xc level)')
    
    plt.tight_layout()
    plt.savefig('trajectory_examples.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Simulation completed: {np.sum(events)} deaths out of {len(events)} individuals")

def example_specific_timepoints():
    """Example with specific time points for trajectory saving."""
    print("Running specific time points example...")
    
    # Create simulation with specific time points
    specific_times = [0, 5, 10, 15, 20, 25, 30, 40, 50]
    sim = getInterventionSR(
        theta,
        n=50,
        t_end=50,
        save_trajectory=True,
        traj_points=specific_times,
        intervention_type='Saturating_eta',
        intervention_effect=[-0.05, 'years'],
        intervention_duration=[5, 15]
    )
    
    # Run simulation
    death_times, events = sim.calc_death_times()
    
    # Plot trajectories
    fig, ax = plt.subplots(figsize=(12, 6))
    sim.plot_trajectories(n_trajectories=8, colormap='tab10', ax=ax)
    ax.set_title('Trajectories at Specific Time Points')
    
    # Add vertical lines for intervention period
    ax.axvspan(5, 15, alpha=0.2, color='red', label='Intervention period')
    ax.legend()
    
    plt.savefig('specific_timepoints_example.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Time points used: {specific_times}")
    print(f"Simulation completed: {np.sum(events)} deaths out of {len(events)} individuals")

def example_multiple_interventions():
    """Compare different intervention types."""
    print("Running multiple interventions comparison...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    intervention_configs = [
        {'type': 'Transient', 'effect': -0.3, 'duration': 10, 'title': 'Transient Intervention'},
        {'type': 'Alpha', 'effect': -0.15, 'duration': [10, 30], 'title': 'Alpha Intervention'},
        {'type': 'Beta change', 'effect': 0.5, 'duration': [10, 30], 'title': 'Beta Change'},
        {'type': 'Damage removal', 'effect': -0.1, 'duration': [10, 30], 'title': 'Damage Removal'}
    ]
    
    for i, config in enumerate(intervention_configs):
        row, col = i // 2, i % 2
        
        sim = getInterventionSR(
            theta,
            n=50,
            t_end=60,
            save_trajectory=True,
            traj_points=120,
            intervention_type=config['type'],
            intervention_effect=config['effect'],
            intervention_duration=config['duration'] if config['type'] != 'Transient' else 0,
            intervention_time=10 if config['type'] == 'Transient' else 0
        )
        
        death_times, events = sim.calc_death_times()
        sim.plot_trajectories(n_trajectories=8, colormap='viridis', ax=axes[row, col])
        axes[row, col].set_title(f"{config['title']} ({np.sum(events)} deaths)")
        
        # Add intervention period visualization
        if config['type'] != 'Transient':
            start, end = config['duration']
            axes[row, col].axvspan(start, end, alpha=0.2, color='red')
        else:
            axes[row, col].axvline(10, alpha=0.5, color='red', linestyle='--')
    
    plt.tight_layout()
    plt.savefig('intervention_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Run examples
    example_basic_plotting()
    example_specific_timepoints()
    example_multiple_interventions()
    
    print("All examples completed! Check the generated PNG files.")
