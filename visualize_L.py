import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
rc('font', family='serif', serif=['Computer Modern Roman'], size=22)
rc('text', usetex=True)

def save_figure(plt, filename, dpi=600):
    """Helper function to save figures in PDF format"""
    plt.savefig(f'{filename}.pdf', format='pdf', dpi=dpi, bbox_inches='tight')

def plot_results(t_test, x_test, mean, variance, save=True):
    plt.figure(figsize=(12, 6))
    
    # Plot true values and predictions
    plt.plot(t_test, x_test[:, 0], 'b-', linewidth=3, label='True')
    plt.plot(t_test, mean[:, 0], 'r--', linewidth=3, label='Predicted')
    
    # Enhanced uncertainty visualization
    std_dev = np.sqrt(variance[:, 0])
    plt.fill_between(t_test, 
                    mean[:, 0] - 2*std_dev,
                    mean[:, 0] + 2*std_dev, 
                    color='r', alpha=0.3, label='2 std interval')
    
    plt.xlabel('Time', fontsize=36)
    plt.ylabel('Temperature (K)', fontsize=36)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.legend(fontsize=25, frameon=False, loc='best')
    plt.grid(True, alpha=0.3)
    
    if save:
        save_figure(plt, 'results_plot_H2')
    plt.show()

def plot_species_comparison(predictions, variances=None, true_values=None, save=True):
    """Plot major and minor species separately"""
    major_indices = [1, 2, 3, 4]  # H2, O2, N2, H2O
    minor_indices = [5, 6, 7, 8, 9]  # H, O, OH, HO2, H2O2
    species_labels = ['H$_2$', 'O$_2$', 'N$_2$', 'H$_2$O', 'H', 'O', 'OH', 'HO$_2$', 'H$_2$O$_2$']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']
    
    # Plot major species
    plt.figure(figsize=(12, 6))
    t = np.arange(len(predictions))

    # Create dummy lines for the line style legend
    plt.plot([], [], 'k--', label='Prediction')
    plt.plot([], [], 'k-', label='True')

    for i, idx in enumerate(major_indices):
        plt.plot(t, predictions[:, idx], '--', linewidth=3, color=colors[i], label=species_labels[i-1])
        if variances is not None:
            std_dev = np.sqrt(variances[:, idx])
            plt.fill_between(t, predictions[:, idx] - 3*std_dev,
                           predictions[:, idx] + 3*std_dev,
                           alpha=0.5, color=colors[i])
        if true_values is not None:
            min_len = min(len(t), len(true_values))
            t_true = np.arange(len(true_values[:min_len]))
            plt.plot(t_true, true_values[:min_len, idx], '-', linewidth=3, color=colors[i])
            
    plt.xlabel('Time-step',fontsize=36)
    plt.ylabel('Mass fraction', fontsize=36)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.grid(True)
    plt.legend(fontsize=25, frameon=False, loc='upper right', ncol=3)
    if save:
        save_figure(plt, 'major_species_comparison_H2')
    plt.show()
    
    # Plot minor species
    plt.figure(figsize=(12, 6))
    
    # Create dummy lines for the line style legend
    plt.plot([], [], 'k--', label='Prediction')
    plt.plot([], [], 'k-', label='True')

    for i, idx in enumerate(minor_indices):
        plt.plot(t, predictions[:, idx], '--', linewidth=3, color=colors[i+4], label=species_labels[i+4])
        if variances is not None:
            std_dev = np.sqrt(variances[:, idx])
            plt.fill_between(t, predictions[:, idx] - 3*std_dev,
                           predictions[:, idx] + 3*std_dev,
                           alpha=0.5, color=colors[i+4])
        if true_values is not None:
            min_len = min(len(t), len(true_values))
            t_true = np.arange(len(true_values[:min_len]))
            plt.plot(t_true, true_values[:min_len, idx], '-', linewidth=3, color=colors[i+4])
    
    plt.xlabel('Time-step', fontsize=36)
    plt.ylabel('Mass fraction', fontsize=36)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.grid(True)
    plt.legend(fontsize=25, frameon=False, loc='upper right', ncol=3)
    if save:
        save_figure(plt, 'minor_species_comparison_H2')
    plt.show()
   
    # Plot mass fraction sum
    plt.figure(figsize=(12, 6))
    species_indices = list(range(1, 10))  # All species indices (1-9)
    mass_sum = np.sum(predictions[:, species_indices], axis=1)
    
    plt.plot(t, mass_sum, 'k-', linewidth=3, label='Sum of Mass Fractions')
    plt.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='Ideal (Sum = 1.0)')
    
    plt.xlabel('Time-step', fontsize=36)
    plt.ylabel('Sum of Mass Fractions', fontsize=36)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.grid(True)
    plt.legend(fontsize=25, frameon=False, loc='best')
    
    if save:
        save_figure(plt, 'mass_sum_conservation')
    plt.show()

def plot_iterative_results(predictions, variances=None, dt=1, true_values=None, save=True):
    t = np.arange(len(predictions))
    n_rows = 4
    n_cols = 3
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 24))
    axes_flat = axes.flatten()
    
    # Labels for each state variable (updated for N2)
    state_labels = ['Temperature [$K$]', 'Y$_{H_2}$', 'Y$_{O_2}$', 'Y$_{N_2}$', 'Y$_{H_2O}$', 
                   'Y$_{H}$', 'Y$_{O}$', 'Y$_{OH}$', 'Y$_{HO_2}$', 'Y$_{H_2O_2}$']
    
    # Plot individual state variables
    for i in range(10):
        if i < len(axes_flat):
            ax = axes_flat[i]
            
            # Plot predictions with thicker line
            ax.plot(t, predictions[:, i], 'r--', linewidth=3, label='Predicted')
            
            if variances is not None:
                std_dev = np.sqrt(variances[:, i])
                # Enhanced uncertainty visualization
                ax.fill_between(t,
                              predictions[:, i] - 2*std_dev,
                              predictions[:, i] + 2*std_dev,
                              color='r', alpha=0.3, label='2 std interval')
            
            if true_values is not None:
                min_len = min(len(t), len(true_values))
                t_true = np.arange(len(true_values[:min_len]))
                ax.plot(t_true, true_values[:min_len, i], 'b-', linewidth=3, label='True')
            
            ax.set_xlabel('Time-step')
            ax.set_ylabel(state_labels[i])
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=25, frameon=False, loc='best')
            
            # Add minor gridlines
            ax.grid(True, which='minor', alpha=0.15)
            ax.minorticks_on()
    
    # Plot mass fraction sum in the last subplot
    if len(axes_flat) > 10:
        ax = axes_flat[10]
        species_indices = list(range(1, 10))  # All species indices (1-9)
        mass_sum = np.sum(predictions[:, species_indices], axis=1)
        
        ax.plot(t, mass_sum, 'k-', linewidth=3, label='Sum of Mass Fractions')
        ax.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='Ideal (Sum = 1.0)')
        
        ax.set_xlabel('Time-step')
        ax.set_ylabel('Sum of Mass Fractions')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=25, frameon=False, loc='best')
        
        # Add minor gridlines
        ax.grid(True, which='minor', alpha=0.15)
        ax.minorticks_on()
    
    # Hide any unused subplots
    for i in range(10, len(axes_flat)):
        if i != 10:  # Skip the mass fraction sum plot
            axes_flat[i].set_visible(False)
            
    plt.tight_layout()
    if save:
        save_figure(plt, 'iterative_results_H2')
    plt.show()

def plot_uncertainty_growth(predictions, variances, dt=1, save=True, scaler=None):
    """
    Visualize uncertainty growth in transformed space.
    
    Args:
        predictions: array of shape (n_steps, n_features) with predictions
        variances: array of shape (n_steps, n_features) with variances
        dt: time step size
        save: whether to save the plots as PDF
        scaler: the StandardScaler or RobustScaler used to transform the data
    """
    if scaler is None:
        raise ValueError("Scaler must be provided to plot transformed uncertainties")
    
    t = np.arange(len(predictions))
    plt.figure(figsize=(12, 6))
    
    # Calculate transformed standard deviations
    std_devs = np.sqrt(variances)
    std_devs_transformed = std_devs / scaler.scale_
    
    # Updated state labels
    state_labels = ['Temp', 'H$_2$', 'O$_2$', 'N$_2$', 'H$_2$O', 'H', 'O', 'OH', 'HO$_2$', 'H$_2$O$_2$']
    
    # Plot transformed uncertainties
    for i in range(10):
        plt.plot(t, std_devs_transformed[:, i], label=state_labels[i], linewidth=3)
    
    plt.xlabel('Time-step', fontsize=36)
    plt.ylabel('$\sigma_{transformed}$', fontsize=36)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.legend(frameon=False, fontsize=25, loc='lower right', ncol=3)
    plt.grid(True)
    plt.yscale('log')
    
    plt.tight_layout()
    if save:
        save_figure(plt, 'uncertainty_growth_H2')
    plt.show()

def plot_scatter_comparison(t_test, x_test, mean, stride=5, save=True):
    """Create scatter plot comparing normalized true vs predicted values."""
    plt.figure(figsize=(8, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    labels = ['T ($K$)', 'H$_2$', 'O$_2$', 'N$_2$', 'H$_2$O', 'H', 'O', 'OH', 'HO$_2$', 'H$_2$O$_2$']
    
    for col in range(10):
        x_scattered = x_test[::stride, col]
        mean_scattered = mean[::stride, col]
        
        x_norm = (x_scattered - x_scattered.min()) / (x_scattered.max() - x_scattered.min())
        mean_norm = (mean_scattered - mean_scattered.min()) / (mean_scattered.max() - mean_scattered.min())
        
        plt.scatter(x_norm, mean_norm, alpha=0.5, color=colors[col], label=labels[col])
    
    plt.plot([0, 1], [0, 1], 'r--', label='True')
    plt.xlabel('Normalised true values', fontsize=32)
    plt.ylabel('Normalised predicted values', fontsize=32)
    plt.xticks(fontsize=27)
    plt.yticks(fontsize=27)
    plt.grid(True)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(frameon=False, fontsize=22, loc='upper left')
    
    if save:
        save_figure(plt, 'scatter_comparison_H2_normalised')
    plt.show()

def plot_training_history(histories, save=True):
    """Plot training histories for all models in the ensemble."""
    plt.figure(figsize=(10, 6))
    for i, history in enumerate(histories):
        plt.plot(history.history['loss'], label=f'Model {i+1}')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(frameon=False, loc='best')
    plt.yscale('log')
    plt.grid(True)
    
    if save:
        save_figure(plt, 'training_history_H2')
    plt.show()
    
def plot_mass_conservation(predictions, save=True):
    """Plot mass conservation over time"""
    t = np.arange(len(predictions))
    
    # Sum all species mass fractions
    species_indices = list(range(1, 10))  # All species (skipping temperature)
    mass_sum = np.sum(predictions[:, species_indices], axis=1)
    
    plt.figure(figsize=(12, 6))
    plt.plot(t, mass_sum, 'b-', linewidth=3, label='Sum of Mass Fractions')
    plt.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='Ideal (Sum = 1.0)')
    
    plt.xlabel('Time-step', fontsize=36)
    plt.ylabel('Sum of Mass Fractions', fontsize=36)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.grid(True)
    plt.legend(fontsize=25, frameon=False, loc='best')
    
    # Calculate and display statistics
    mean_error = np.mean(np.abs(mass_sum - 1.0))
    plt.title(f'Mass Conservation Error: {mean_error:.4f}', fontsize=36)
    
    if save:
        save_figure(plt, 'mass_conservation')
    plt.show()
    
    return mean_error