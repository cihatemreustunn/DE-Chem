from data_preprocessing_L import data_import, save_transform_info
from deep_ensemble_L import DeepEnsemble
import tensorflow as tf

def train_model(learning_rate=0.001, n_models=5, train_model=True,
                use_mass_constraint=False, mass_constraint_weight=0.0,
                mse_weight=0.7, phased_training=False, phase_split=0.5,
                use_log_transform=True, epsilon=1e-10):
    """
    Train the model with hybrid loss function and optional mass constraint
    
    Args:
        learning_rate: Learning rate for the optimizer
        n_models: Number of models in the ensemble
        train_model: Whether to train the model or just return the data
        use_mass_constraint: Whether to include mass constraint loss
        mass_constraint_weight: Weight for the mass constraint term
        mse_weight: Weight for MSE component in hybrid loss (0-1)
        phased_training: Whether to use phased training
        phase_split: Fraction of epochs for phase 1 (MSE only) in phased training
        use_log_transform: Whether to apply log transformation to mass fractions
        epsilon: Small value added to mass fractions to avoid log(0)
        
    Returns:
        ensemble, data_tuple, histories, scaler, transform_info
    """
    # Import data with optional log transformation
    X_train, X_val, Y_train, Y_val, X_test, Y_test, scaler, transform_info = data_import(
        use_log_transform=use_log_transform, 
        epsilon=epsilon
    )
    
    if not train_model:
        return None, (X_val, Y_val, X_test, Y_test), None, scaler, transform_info
    
    # Save transformation info for future use
    save_transform_info(transform_info)
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).shuffle(1004000).batch(2048).prefetch(tf.data.AUTOTUNE)
    
    # Create and compile ensemble
    ensemble = DeepEnsemble(n_models=n_models, learning_rate=learning_rate)
    ensemble.compile(
        use_mass_constraint=use_mass_constraint, 
        mass_constraint_weight=mass_constraint_weight,
        mse_weight=mse_weight,
        scaler=scaler,
        transform_info=transform_info
    )
    
    # Fit ensemble
    histories = ensemble.fit(
        dataset, 
        phased_training=phased_training, 
        phase_split=phase_split
    )
    
    return ensemble, (X_val, Y_val, X_test, Y_test), histories, scaler, transform_info