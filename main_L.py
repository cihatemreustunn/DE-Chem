from train_L import train_model
from evaluate_L import evaluate_model
from eval_function import evaluate_model_comprehensive
from compare_metrics import evaluate_model_comprehensive_H2, plot_metrics_comparison
from visualize_L import plot_results, plot_iterative_results, plot_training_history
from visualize_L import plot_uncertainty_growth, plot_scatter_comparison, plot_species_comparison
from visualize_L import plot_mass_conservation
from deep_ensemble_L import DeepEnsemble
from data_preprocessing_L import inverse_transform_predictions, load_transform_info
import numpy as np
import os
import time
import tensorflow as tf
import matplotlib.pyplot as plt

# Configure TensorFlow for better CPU performance
tf.config.threading.set_intra_op_parallelism_threads(7)
tf.config.threading.set_inter_op_parallelism_threads(7)

def main():
    # Training parameters
    learning_rate = 0.001
    n_models = 2
    model_dir = 'BigData_0.01std_2NNs_ver6'
    
    # Mass conservation parameters
    use_mass_constraint = False
    mass_constraint_weight = 0.1  # Reduced from 0.5 to give more emphasis to dynamics
    
    # Hybrid loss parameters
    mse_weight = 0.7 # 70% MSE, 30% NLL
    
    # Phased training parameters
    phased_training = False
    phase_split = 0.9  # 70% of epochs for hybrid loss only, 30% for hybrid + mass constraint
    
    # Log transformation parameters
    use_log_transform = True
    epsilon = 1e-10  # Small value added to mass fractions to avoid log(0)

    if os.path.exists(model_dir):
        print("Loading existing ensemble...")
        ensemble = DeepEnsemble.load_models(model_dir)
        transform_info = ensemble.transform_info
        _, (X_val, Y_val, X_test, Y_test), _, scaler, _ = train_model(
            learning_rate=learning_rate,
            n_models=n_models,
            train_model=False,
            use_log_transform=use_log_transform,
            epsilon=epsilon
        )
        histories = None
    else:
        print(f"Training new ensemble with hybrid loss (MSE weight: {mse_weight}, mass constraint: {use_mass_constraint})")
        ensemble, (X_val, Y_val, X_test, Y_test), histories, scaler, transform_info = train_model(
            learning_rate=learning_rate,
            n_models=n_models,
            use_mass_constraint=use_mass_constraint,
            mass_constraint_weight=mass_constraint_weight,
            mse_weight=mse_weight,
            phased_training=phased_training, 
            phase_split=phase_split,
            use_log_transform=use_log_transform,
            epsilon=epsilon
        )
        ensemble.save_models(model_dir)

    if histories is not None:
        # Plot the standard training history
        plot_training_history(histories)

    # Pre-compile models for faster inference
    print("\nCompiling models...")
    ensemble.compile_models()

    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_pred_mean, val_pred_var = ensemble.predict(X_val)
    
    # Convert predictions back to original scale
    val_pred_unscaled = inverse_transform_predictions(val_pred_mean, scaler, transform_info)
    Y_val_unscaled = inverse_transform_predictions(Y_val, scaler, transform_info)
    
    # Calculate MSE
    val_mse = np.mean((Y_val_unscaled - val_pred_unscaled)**2)
    val_uncertainty = np.mean(np.sqrt(val_pred_var))
    
    print(f'Validation MSE: {val_mse:.6f}')
    print(f'Validation Mean Uncertainty: {val_uncertainty:.6f}')

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_pred_mean, test_pred_var = ensemble.predict(X_test)
    
    # Convert predictions back to original scale
    test_pred_unscaled = inverse_transform_predictions(test_pred_mean, scaler, transform_info)
    Y_test_unscaled = inverse_transform_predictions(Y_test, scaler, transform_info)
    
    # Calculate MSE
    test_mse = np.mean((Y_test_unscaled - test_pred_unscaled)**2)
    test_uncertainty = np.mean(np.sqrt(test_pred_var))
    
    print(f'Test MSE: {test_mse:.6f}')
    print(f'Test Mean Uncertainty: {test_uncertainty:.6f}')

    # Make iterative predictions
    n_steps = 250
    batch_size = 50
    print("\nMaking iterative predictions...")
    
    # Warm-up run
    _ = ensemble.predict_iterative(X_test[0], 10, scaler, batch_size=batch_size, 
                              transform_info=transform_info, accumulate_variance=False)
    
    # Actual timed run
    start_time = time.time()
    predictions, variances = ensemble.predict_iterative(
        X_test[0], n_steps, scaler, 
        batch_size=batch_size, 
        transform_info=transform_info,
        accumulate_variance=False
    )
    end_time = time.time()
    
    inference_time = end_time - start_time
    print(f"Total inference time for {n_steps} steps: {inference_time:.4f} seconds")
    print(f"Average time per step: {(inference_time/n_steps)*1000:.4f} ms")
    
    # Calculate mass fraction sum error on predictions
    species_idx = slice(1, 10)  # Indices 1-9 (all except temperature)
    mass_sum = np.sum(predictions[:, species_idx], axis=1)
    mass_error = np.abs(mass_sum - 1.0).mean()
    print(f"\nMass Fraction Sum Error: {mass_error:.6f}")
    print(f"Mass Fraction Sum Mean: {mass_sum.mean():.6f}")
    print(f"Mass Fraction Sum Min: {mass_sum.min():.6f}")
    print(f"Mass Fraction Sum Max: {mass_sum.max():.6f}")
    
    # Plot mass conservation
    mass_error = plot_mass_conservation(predictions)
    print(f"Mass Conservation Error (from plot): {mass_error:.6f}")
    
    # Additional visualizations
    print("\nGenerating visualizations...")
    plot_uncertainty_growth(predictions, variances, dt=1, save=True, scaler=scaler)
    plot_iterative_results(predictions, variances, dt=1, true_values=Y_test_unscaled)
    plot_species_comparison(predictions, variances, true_values=Y_test_unscaled)
    
    t_test = np.arange(len(Y_test))
    plot_scatter_comparison(t_test, Y_test_unscaled, test_pred_unscaled, stride=5)
    
    print("\nDone!")

if __name__ == "__main__":
    main()