import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats
import pandas as pd
import os
from datetime import datetime

def evaluate_model_comprehensive(model, X_heldout, Y_heldout, scaler, save_dir='evaluation_results'):
    """
    Comprehensive model evaluation with metrics saved to CSV files.
    """
    mean, variance = model.predict(X_heldout)
    
    # Inverse transform predictions and true values
    mean_unscaled = scaler.inverse_transform(mean)
    Y_unscaled = scaler.inverse_transform(Y_heldout)
    variance_unscaled = variance * (scaler.scale_**2)
    
    # Calculate overall metrics
    overall_metrics = {
        'mse': mean_squared_error(Y_unscaled[:, 1:], mean_unscaled[:, 1:]),
        'rmse': np.sqrt(mean_squared_error(Y_unscaled[:, 1:], mean_unscaled[:, 1:])),
        'mae': mean_absolute_error(Y_unscaled[:, 1:], mean_unscaled[:, 1:]),
        'r2': r2_score(Y_unscaled[:, 1:], mean_unscaled[:, 1:]),
        'mean_uncertainty': np.mean(np.sqrt(variance_unscaled))
    }
    
    # Calculate per-variable metrics
    per_variable_metrics = []
    state_labels = ['Temperature', 'Y_H2', 'Y_O2', 'Y_H2O', 'Y_H', 'Y_O', 'Y_OH', 'Y_HO2', 'Y_H2O2']
    
    for i in range(Y_unscaled.shape[1]):
        metrics = {
            'variable': state_labels[i],
            'mse': mean_squared_error(Y_unscaled[:, i], mean_unscaled[:, i]),
            'rmse': np.sqrt(mean_squared_error(Y_unscaled[:, i], mean_unscaled[:, i])),
            'mae': mean_absolute_error(Y_unscaled[:, i], mean_unscaled[:, i]),
            'r2': r2_score(Y_unscaled[:, i], mean_unscaled[:, i]),
            'mean_uncertainty': np.mean(np.sqrt(variance_unscaled[:, i])),
            'correlation': stats.pearsonr(Y_unscaled[:, i], mean_unscaled[:, i])[0]
        }
        per_variable_metrics.append(metrics)
    
    # Save results to CSV
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save overall metrics
    overall_df = pd.DataFrame([overall_metrics])
    overall_df.to_csv(f'{save_dir}/overall_metrics_{timestamp}.csv', index=False)
    
    # Save per-variable metrics
    var_metrics_df = pd.DataFrame(per_variable_metrics)
    var_metrics_df.to_csv(f'{save_dir}/per_variable_metrics_{timestamp}.csv', index=False)
    
    return overall_metrics, per_variable_metrics, mean_unscaled, variance_unscaled