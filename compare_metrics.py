import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats
import pandas as pd
import os
from datetime import datetime

def evaluate_model_comprehensive_H2(model, X_heldout, Y_heldout, scaler):
    mean, variance = model.predict(X_heldout)
    mean_unscaled = scaler.inverse_transform(mean)
    Y_unscaled = scaler.inverse_transform(Y_heldout)
    variance_unscaled = variance * (scaler.scale_**2)
    
    per_variable_metrics = []
    state_labels = ['Temperature', 'H$_2$', 'O$_2$', 'H$_2$O', 'H', 'O', 'OH', 'HO$_2$', 'H$_2$O$_2$']
    
    for i in range(Y_unscaled.shape[1]):
        metrics = {
            'variable': state_labels[i],
            'mse': mean_squared_error(Y_unscaled[:, i], mean_unscaled[:, i]),
            'mae': mean_absolute_error(Y_unscaled[:, i], mean_unscaled[:, i]),
            'r2': r2_score(Y_unscaled[:, i], mean_unscaled[:, i]),
            'mean_uncertainty': np.mean(np.sqrt(variance_unscaled[:, i]))
        }
        per_variable_metrics.append(metrics)
    
    return per_variable_metrics

def plot_metrics_comparison(val_metrics, test_metrics):
    species_labels = ['H$_2$', 'O$_2$', 'H$_2$O', 'H', 'O', 'OH', 'HO$_2$', 'H$_2$O$_2$']
    
    val_r2 = [m['r2'] for m in val_metrics[1:]]
    test_r2 = [m['r2'] for m in test_metrics[1:]]
    
    val_mae = [m['mae'] for m in val_metrics[1:]]
    test_mae = [m['mae'] for m in test_metrics[1:]]
    
    x = np.arange(len(species_labels))
    width = 0.35
    
    # R2 plot
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, val_r2, width, label='Validation', color='blue')
    plt.bar(x + width/2, test_r2, width, label='Test', color='red')
    plt.ylabel('R$^2$ Score', fontsize=35)
    plt.xlabel('Species', fontsize=35)
    plt.ylim(0.8, 1)
    plt.yticks(fontsize=28)
    plt.xticks(x, species_labels, rotation=45, ha='right', fontsize=28)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('r2_comparison_H2.pdf', bbox_inches='tight')
    plt.show()
    
    # MAE plot
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, val_mae, width, label='Validation', color='blue')
    plt.bar(x + width/2, test_mae, width, label='Test', color='red')
    plt.ylabel('MAE', fontsize=30)
    plt.xlabel('Species', fontsize=30)
    plt.yscale('log')
    plt.yticks(fontsize=28)
    plt.xticks(x, species_labels, rotation=45, ha='right', fontsize=28)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('mae_comparison_H2.pdf', bbox_inches='tight')
    plt.show()