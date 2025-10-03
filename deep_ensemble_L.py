import numpy as np
from base_model_L import BaseModel
import tensorflow as tf
import os
import pickle

# Define custom hybrid loss function
def create_hybrid_loss(scaler, transform_info, mse_weight=0.7, mass_constraint_weight=0.3, 
                      use_mass_constraint=True):
    """Creates a hybrid loss function"""
    
    if scaler is None:
        raise ValueError("Scaler must be provided")
    
    # Convert scaler parameters to TensorFlow constants
    scale = tf.constant(scaler.scale_, dtype=tf.float32)
    mean = tf.constant(scaler.mean_, dtype=tf.float32)
    
    # Get transformation info
    use_log = transform_info and transform_info.get('use_log_transform', False)
    epsilon = transform_info.get('epsilon', 1e-10) if use_log else 0
    species_indices = tf.constant(transform_info.get('species_indices', list(range(1, 10))), dtype=tf.int32)
    
    def hybrid_loss(y_true, y_pred):
        """Hybrid loss function"""
        
        # Split predictions into mean and log variance
        if isinstance(y_pred, tuple):
            mean_pred, log_var_pred = y_pred
        else:
            mean_pred = y_pred
            log_var_pred = tf.zeros_like(mean_pred)
            
        # MSE component for accurate dynamics
        mse_loss = tf.reduce_mean(tf.square(y_true - mean_pred))
        
        # NLL component for uncertainty
        variance = tf.exp(log_var_pred) + 1e-6
        nll_loss = 0.5 * tf.reduce_mean(log_var_pred + tf.square(y_true - mean_pred) / variance)
        
        # Combined MSE and NLL
        combined_loss = mse_weight * mse_loss + (1 - mse_weight) * nll_loss
        
        # Initialize mass error
        mass_error = tf.constant(0.0, dtype=tf.float32)
        
        # Mass conservation constraint (optional)
        if use_mass_constraint:
            # Convert to unscaled space for mass fraction calculation
            mean_pred_unscaled = mean_pred * scale + mean
            
            # Extract species values (skip temperature at index 0)
            species_pred = tf.gather(mean_pred_unscaled, species_indices, axis=1)
            
            if use_log:
                # If using log transformation, convert back to linear space
                species_pred = tf.exp(species_pred) - epsilon
                # Ensure values are positive
                species_pred = tf.maximum(species_pred, 0.0)
            
            # Calculate sum of mass fractions
            sum_pred = tf.reduce_sum(species_pred, axis=1)
            
            # Calculate mass fraction error (should be close to 1.0)
            mass_error = tf.reduce_mean(tf.abs(sum_pred - 1.0))
            
            # Add mass constraint to loss
            combined_loss += mass_constraint_weight * mass_error
        
        return combined_loss
    
    return hybrid_loss

def calculate_improved_uncertainty(means, log_vars, scaler, transform_info=None):
    """
    Calculate improved uncertainty estimates that are more physically plausible
    
    Args:
        means: Predicted means from all ensemble members [n_models, n_features] or [n_models, batch, n_features]
        log_vars: Log variances from all ensemble models [n_models, n_features] or [n_models, batch, n_features]
        scaler: The scaler used for normalization
        transform_info: Information about log transformations
        
    Returns:
        mean: Ensemble mean prediction
        variance: Improved variance estimate
    """
    # Convert to numpy if tensors
    means_np = np.array([m.numpy() if hasattr(m, 'numpy') else m for m in means])
    log_vars_np = np.array([lv.numpy() if hasattr(lv, 'numpy') else lv for lv in log_vars])
    
    # Ensure 2D arrays for consistent processing
    if means_np.ndim == 2:  # [n_models, n_features]
        pass
    elif means_np.ndim == 3:  # [n_models, batch, n_features]
        # Take the first batch element for consistency
        means_np = means_np[:, 0, :]
        log_vars_np = log_vars_np[:, 0, :]
    else:
        raise ValueError(f"Unexpected shape for means: {means_np.shape}")
    
    # Calculate ensemble mean
    mean = np.mean(means_np, axis=0)  # shape: [n_features]
    
    # Convert log variances to variances
    variances = np.exp(log_vars_np)
    
    # Calculate aleatoric uncertainty (average model variance)
    aleatoric_variance = np.mean(variances, axis=0)  # shape: [n_features]
    
    # Calculate epistemic uncertainty (variance of means)
    # For numerical stability with small ensembles, use a minimum of 2 models
    if means_np.shape[0] >= 2:
        epistemic_variance = np.var(means_np, axis=0, ddof=1)  # Use unbiased estimator
    else:
        epistemic_variance = np.zeros_like(aleatoric_variance)
    
    # Apply a weighting factor to balance aleatoric and epistemic uncertainty
    # This helps prevent double-counting
    alpha = 0.7  # Weight for aleatoric uncertainty (tune this parameter)
    combined_variance = alpha * aleatoric_variance + (1 - alpha) * epistemic_variance
    
    # Transform back to original scale
    if transform_info and transform_info.get('use_log_transform', True):
        epsilon = transform_info.get('epsilon', 1e-10)
        species_indices = transform_info.get('species_indices', list(range(1, len(mean))))
        
        # Transform mean and variance back to original scale
        mean_unscaled = np.zeros(mean.shape)
        variance_unscaled = np.zeros(combined_variance.shape)
        
        # First transform back to original scale (un-standardize)
        mean_scaled = mean * scaler.scale_ + scaler.mean_
        variance_scaled = combined_variance * (scaler.scale_**2)
        
        # Perform inverse transform with physical constraints
        for i in range(len(mean)):
            if i in species_indices:  # For mass fractions (log-transformed)
                # Apply bounded delta method for log-normal transformation
                # This prevents extreme uncertainty for small concentrations
                exp_mean = np.exp(mean_scaled[i])
                
                # Apply a dampening factor for very small concentrations to avoid explosion
                # This caps the relative uncertainty to a maximum value
                rel_std = np.sqrt(variance_scaled[i])
                max_rel_std = 2.0  # Maximum relative std dev (tune this parameter)
                if rel_std > max_rel_std:
                    rel_std = max_rel_std
                
                # Calculate variance in original space with damping for numerical stability
                variance_unscaled[i] = (exp_mean * rel_std)**2
                
                # Transform mean
                mean_unscaled[i] = exp_mean - epsilon
                mean_unscaled[i] = max(mean_unscaled[i], 0.0)  # Ensure non-negative
            else:  # For temperature (not log-transformed)
                mean_unscaled[i] = mean_scaled[i]
                variance_unscaled[i] = variance_scaled[i]
        
        # Apply physical constraints to uncertainties
        # Mass fractions can't be negative and shouldn't exceed 1
        for i in species_indices:
            # Limit standard deviation to physically meaningful bounds
            std_dev = np.sqrt(variance_unscaled[i])
            # For mass fractions, std dev shouldn't be larger than the mean or 1-mean
            if mean_unscaled[i] > 0.01:  # For non-trace species
                max_std = min(mean_unscaled[i] * 0.5, (1.0 - mean_unscaled[i]) * 0.5)
                std_dev = min(std_dev, max_std)
                variance_unscaled[i] = std_dev**2
            else:  # For trace species, use relative uncertainty with a cap
                max_rel_std = min(2.0, 1e-5 / max(mean_unscaled[i], 1e-10))
                variance_unscaled[i] = (mean_unscaled[i] * max_rel_std)**2
        
        return mean_unscaled, variance_unscaled
    else:
        # For non-transformed data, just apply scaling
        mean_unscaled = scaler.inverse_transform(mean.reshape(1, -1))[0]
        variance_unscaled = combined_variance * (scaler.scale_**2)
        return mean_unscaled, variance_unscaled

class DeepEnsemble:
    def __init__(self, n_models=5, model_dir='BaseModel', learning_rate=0.001):
        self.n_models = n_models
        self.learning_rate = learning_rate
        self.models = [BaseModel(learning_rate=learning_rate, seed=42+i) for i in range(n_models)]
        self.model_dir = model_dir
        self.state_dim = 10  # 10 for the dataset with N2
        self.combined_model = None
        self.use_mass_constraint = False
        self.mass_constraint_weight = 0.0
        self.mse_weight = 0.7  # Default weight for MSE in hybrid loss
        self.scaler = None
        self.transform_info = None

    def set_scaler(self, scaler):
        """Store the scaler for use in mass fraction calculations"""
        self.scaler = scaler
        
    def set_transform_info(self, transform_info):
        """Store the transform information for use in mass fraction calculations"""
        self.transform_info = transform_info

    def create_hybrid_loss(self):
        """Creates the hybrid loss function with current settings"""
        return create_hybrid_loss(
            self.scaler, 
            self.transform_info,
            self.mse_weight,
            self.mass_constraint_weight,
            self.use_mass_constraint
        )

    def compile(self, loss='mse', use_mass_constraint=False, mass_constraint_weight=0.0, 
                mse_weight=0.7, scaler=None, transform_info=None):
        """Compile models with the specified loss function"""
        if scaler is not None:
            self.set_scaler(scaler)
            
        if transform_info is not None:
            self.set_transform_info(transform_info)
            
        self.use_mass_constraint = use_mass_constraint
        self.mass_constraint_weight = mass_constraint_weight
        self.mse_weight = mse_weight
        
        if self.scaler is None:
            raise ValueError("Scaler must be provided")
        if self.transform_info is None:
            raise ValueError("Transform info must be provided")
                
        # Create loss function
        hybrid_loss = self.create_hybrid_loss()
        
        for model in self.models:
            model.compile(optimizer=model.get_optimizer(), loss=hybrid_loss)

    def create_combined_model(self):
        """Create a combined model for inference that outputs mean and variance"""
        inputs = tf.keras.Input(shape=(self.state_dim,))
        
        # Collect means and log variances from all models
        means = []
        log_vars = []
        
        for model in self.models:
            mean, log_var = model(inputs)
            means.append(mean)
            log_vars.append(log_var)
            
        # Stack and compute ensemble mean and variance
        stacked_means = tf.stack(means, axis=0)
        stacked_log_vars = tf.stack(log_vars, axis=0)
        
        # Mean of the predicted means
        ensemble_mean = tf.reduce_mean(stacked_means, axis=0)
        
        # For variance, we need to account for:
        # 1. Variance within each model (aleatoric uncertainty)
        # 2. Variance between model predictions (epistemic uncertainty)
        
        # Convert log variances to variances
        model_variances = tf.exp(stacked_log_vars)
        
        # 1. Average of the model variances (aleatoric)
        avg_model_variance = tf.reduce_mean(model_variances, axis=0)
        
        # 2. Variance of the means (epistemic)
        variance_of_means = tf.math.reduce_variance(stacked_means, axis=0)
        
        # Total predictive variance = aleatoric + epistemic
        ensemble_variance = avg_model_variance + variance_of_means
        
        # Convert back to log variance for consistency
        ensemble_log_var = tf.math.log(ensemble_variance)
        
        self.combined_model = tf.keras.Model(
            inputs=inputs,
            outputs=[ensemble_mean, ensemble_variance]
        )

    def compile_models(self):
        """Compile the combined model for inference"""
        self.create_combined_model()
        sample_input = np.zeros((1, self.state_dim))
        _ = self.combined_model(sample_input)

    def fit(self, dataset, epochs=200, verbose=1, 
            callbacks=None, phased_training=False, phase_split=0.5):
        """Train the ensemble models"""
        histories = []
    
        print("\nModel Architecture:")
        sample_input = tf.keras.Input(shape=(self.state_dim,))
        self.models[0](sample_input)
        self.models[0].summary()
        print(f"\nTraining {self.n_models} models with this architecture\n")
        
        # Default callbacks list if none provided
        if callbacks is None:
            callbacks = [tf.keras.callbacks.EarlyStopping(monitor='loss', patience=25)]
    
        if phased_training and self.use_mass_constraint:
            # Phase 1: Train with only hybrid loss (no mass constraint)
            original_weight = self.mass_constraint_weight
            self.mass_constraint_weight = 0.0
            phase1_epochs = int(epochs * phase_split)
            
            print(f"\nPhase 1: Training with hybrid loss (MSE weight: {self.mse_weight}) for {phase1_epochs} epochs")
            phase1_histories = []
            
            for i, model in enumerate(self.models):                
                print(f'Training model {i+1}/{self.n_models} (learning rate: {self.learning_rate})')
                
                # Create a loss function
                self.compile(use_mass_constraint=False, mse_weight=self.mse_weight)
                
                # Fit model
                history1 = model.fit(dataset, epochs=phase1_epochs, verbose=verbose, 
                                    callbacks=callbacks)
                phase1_histories.append(history1)
                
            # Phase 2: Continue training with mass constraint
            self.mass_constraint_weight = original_weight
            phase2_epochs = epochs - phase1_epochs
            print(f"\nPhase 2: Training with hybrid loss + Mass Constraint (weight: {self.mass_constraint_weight}) for {phase2_epochs} epochs")
            
            for i, model in enumerate(self.models):                
                print(f'Continuing model {i+1}/{self.n_models}')
                
                # Create a loss function with mass constraint
                self.compile(use_mass_constraint=True, 
                            mass_constraint_weight=self.mass_constraint_weight,
                            mse_weight=self.mse_weight)
                
                # Fit model
                history2 = model.fit(dataset, epochs=phase2_epochs, verbose=verbose, 
                                    callbacks=callbacks)
                
                # Combine histories for this model
                combined_history = tf.keras.callbacks.History()
                combined_history.epoch = phase1_histories[i].epoch + history2.epoch
                combined_history.history = {
                    k: phase1_histories[i].history.get(k, []) + history2.history.get(k, []) 
                    for k in set(phase1_histories[i].history) | set(history2.history)
                }
                histories.append(combined_history)
                
        else:
            # Normal training
            for i, model in enumerate(self.models):                
                print(f'\nTraining model {i+1}/{self.n_models} (learning rate: {self.learning_rate})')
                
                # Create a loss function
                self.compile(use_mass_constraint=self.use_mass_constraint,
                            mass_constraint_weight=self.mass_constraint_weight,
                            mse_weight=self.mse_weight)
                
                # Fit model
                history = model.fit(dataset, epochs=epochs, verbose=verbose, 
                                   callbacks=callbacks)
                histories.append(history)
        
        return histories

    def predict(self, X):
        """
        Make predictions with the ensemble
        
        Args:
            X: Input data
            
        Returns:
            means: Predicted means
            variances: Predicted variances
        """
        if self.combined_model is None:
            self.create_combined_model()
            
        # Get predictions from all models
        means = []
        variances = []
        
        for model in self.models:
            mean, log_var = model(X)
            means.append(mean)
            variances.append(tf.exp(log_var))
            
        # Stack predictions
        stacked_means = np.stack(means, axis=0)
        stacked_variances = np.stack(variances, axis=0)
        
        # Compute ensemble mean and variance
        ensemble_mean = np.mean(stacked_means, axis=0)
        
        # Total variance = average of variances + variance of means
        avg_variance = np.mean(stacked_variances, axis=0)
        variance_of_means = np.var(stacked_means, axis=0)
        ensemble_variance = avg_variance + variance_of_means
        
        return ensemble_mean, ensemble_variance

    def predict_iterative(self, initial_state, n_steps, scaler, batch_size=50, 
                         include_variance=True, transform_info=None, accumulate_variance=False):
        """
        Make iterative predictions with improved uncertainty estimates
    
        Args:
            initial_state: Initial state for prediction
            n_steps: Number of steps to predict
            scaler: Data scaler for transformations
            batch_size: Batch size for making predictions
            include_variance: Whether to include variance in output
            transform_info: Information about transformations
            accumulate_variance: Whether to accumulate variance over time steps (True)
                                or use independent variances at each step (False)
        
        Returns:
            predictions_unscaled: Predictions in original scale
            variances_unscaled: Variances in original scale (if include_variance=True)
        """
        if initial_state.shape != (self.state_dim,):
            raise ValueError(f"Initial state must have shape ({self.state_dim},), got {initial_state.shape}")

        predictions_unscaled = np.zeros((n_steps, self.state_dim))
        variances_unscaled = np.zeros((n_steps, self.state_dim))
    
        # Scaled predictions for next step inputs
        current_state = initial_state.reshape(1, -1)

        for step in range(0, n_steps, batch_size):
            end_step = min(step + batch_size, n_steps)
        
            current_batch_state = current_state
    
            for i in range(end_step - step):
                # Get predictions from all models
                all_means = []
                all_log_vars = []
            
                for model in self.models:
                    mean, log_var = model(current_batch_state)
                    all_means.append(mean)
                    all_log_vars.append(log_var)
            
                # Calculate improved uncertainty
                step_idx = step + i
            
                # Get mean for next time step input (in scaled space)
                mean_scaled = np.mean([m.numpy() for m in all_means], axis=0)
            
                # Calculate unscaled predictions with improved uncertainty
                mean_unscaled, variance_unscaled = calculate_improved_uncertainty(
                    all_means, all_log_vars, scaler, transform_info
                )
            
                # Store predictions
                predictions_unscaled[step_idx] = mean_unscaled
            
                # For variance, handle accumulation if requested
                if accumulate_variance and step_idx > 0:
                    # Simple linear accumulation of variance with damping
                    damping_factor = 0.9  # Prevents excessive growth
                    variances_unscaled[step_idx] = variance_unscaled + damping_factor * variances_unscaled[step_idx-1]
                else:
                    variances_unscaled[step_idx] = variance_unscaled
            
                # Use scaled mean prediction for next step
                current_batch_state = mean_scaled.reshape(1, -1)
    
            # Update current state for next batch
            current_state = current_batch_state

        if include_variance:
            return predictions_unscaled, variances_unscaled
        return predictions_unscaled


    def save_models(self, custom_dir=None):
        """
        Save the ensemble models to disk
        
        Args:
            custom_dir: Directory to save models (default is self.model_dir)
        """
        save_dir = custom_dir if custom_dir else self.model_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Create custom objects dictionary for saving
        custom_objects = {
            'hybrid_loss': self.create_hybrid_loss()
        }
        
        for i, model in enumerate(self.models):
            model_path = os.path.join(save_dir, f'model_{i}')
            model.save(model_path, save_format='tf')
        
        metadata = {
            'n_models': self.n_models,
            'model_dir': save_dir,
            'state_dim': self.state_dim,
            'learning_rate': self.learning_rate,
            'use_mass_constraint': self.use_mass_constraint,
            'mass_constraint_weight': self.mass_constraint_weight,
            'mse_weight': self.mse_weight
        }
        np.save(os.path.join(save_dir, 'metadata.npy'), metadata)
        
        # Save scaler for consistent calculations
        if self.scaler is not None:
            with open(os.path.join(save_dir, 'scaler.pkl'), 'wb') as f:
                pickle.dump(self.scaler, f)
                
        # Save transform_info if available
        if self.transform_info is not None:
            with open(os.path.join(save_dir, 'transform_info.pkl'), 'wb') as f:
                pickle.dump(self.transform_info, f)
                
        print(f'Ensemble saved to {save_dir}')

    @classmethod
    def load_models(cls, model_dir):
        """
        Load ensemble models from disk
        
        Args:
            model_dir: Directory containing saved models
            
        Returns:
            ensemble: Loaded DeepEnsemble object
        """
        if not os.path.exists(model_dir):
            raise ValueError(f"Directory {model_dir} does not exist")
            
        metadata = np.load(os.path.join(model_dir, 'metadata.npy'), allow_pickle=True).item()
        ensemble = cls(
            n_models=metadata['n_models'], 
            model_dir=model_dir,
            learning_rate=metadata.get('learning_rate', 0.001)
        )
        
        # Set state_dim from metadata
        ensemble.state_dim = metadata.get('state_dim', 10)
        
        # Load scaler if available
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                ensemble.scaler = pickle.load(f)
        
        # Load transform_info if available
        transform_path = os.path.join(model_dir, 'transform_info.pkl')
        if os.path.exists(transform_path):
            with open(transform_path, 'rb') as f:
                ensemble.transform_info = pickle.load(f)
        
        # Load mass constraint settings
        ensemble.use_mass_constraint = metadata.get('use_mass_constraint', False)
        ensemble.mass_constraint_weight = metadata.get('mass_constraint_weight', 0.0)
        ensemble.mse_weight = metadata.get('mse_weight', 0.7)
        
        # Create custom objects dictionary for loading
        if ensemble.scaler is not None and ensemble.transform_info is not None:
            custom_loss = create_hybrid_loss(
                ensemble.scaler, 
                ensemble.transform_info,
                ensemble.mse_weight,
                ensemble.mass_constraint_weight,
                ensemble.use_mass_constraint
            )
            custom_objects = {'hybrid_loss': custom_loss}
        else:
            custom_objects = None
        
        # Load models with custom objects
        ensemble.models = []
        for i in range(metadata['n_models']):
            model_path = os.path.join(model_dir, f'model_{i}')
            
            try:
                model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
                ensemble.models.append(model)
            except Exception as e:
                print(f"Error loading model {i}: {e}")
                # Try loading model without custom objects
                model = tf.keras.models.load_model(model_path)
                ensemble.models.append(model)
                
        print(f'Loaded ensemble from {model_dir}')
        return ensemble

    # Add methods to manually compute and print loss components
    def compute_loss_components(self, X, y):
        """
        Manually compute and print loss components for a given batch
        
        Args:
            X: Input data
            y: Target data
            
        Returns:
            mse_loss, nll_loss, mass_loss, total_loss
        """
        # Ensure TensorFlow eager execution
        with tf.GradientTape() as tape:
            mse_losses = []
            nll_losses = []
            mass_losses = []
            total_losses = []
            
            # Process batch for each model
            for i, model in enumerate(self.models):
                mean_pred, log_var_pred = model(X)
                
                # MSE component
                mse_loss = tf.reduce_mean(tf.square(y - mean_pred)).numpy()
                
                # NLL component
                variance = tf.exp(log_var_pred) + 1e-6
                nll_loss = 0.5 * tf.reduce_mean(log_var_pred + tf.square(y - mean_pred) / variance).numpy()
                
                # Mass conservation component
                mean_pred_unscaled = mean_pred * self.scaler.scale_ + self.scaler.mean_
                species_indices = self.transform_info.get('species_indices', list(range(1, 10)))
                species_pred = mean_pred_unscaled[:, species_indices]
                
                # Handle log transform if used
                if self.transform_info and self.transform_info.get('use_log_transform', False):
                    epsilon = self.transform_info.get('epsilon', 1e-10)
                    species_pred = np.exp(species_pred) - epsilon
                    # Ensure values are positive
                    species_pred = np.maximum(species_pred, 0.0)
                
                # Calculate sum of mass fractions
                sum_pred = np.sum(species_pred, axis=1)
                mass_loss = np.mean(np.abs(sum_pred - 1.0))
                
                # Total loss
                total_loss = self.mse_weight * mse_loss + (1 - self.mse_weight) * nll_loss
                if self.use_mass_constraint:
                    total_loss += self.mass_constraint_weight * mass_loss
                    
                mse_losses.append(mse_loss)
                nll_losses.append(nll_loss)
                mass_losses.append(mass_loss)
                total_losses.append(total_loss)
                
                print(f"Model {i+1} - MSE: {mse_loss:.6f}, NLL: {nll_loss:.6f}, "
                     f"Mass: {mass_loss:.6f}, Total: {total_loss:.6f}")
            
            # Average across models
            avg_mse = np.mean(mse_losses)
            avg_nll = np.mean(nll_losses)
            avg_mass = np.mean(mass_losses) 
            avg_total = np.mean(total_losses)
            
            print(f"\nAverage - MSE: {avg_mse:.6f}, NLL: {avg_nll:.6f}, "
                 f"Mass: {avg_mass:.6f}, Total: {avg_total:.6f}")
            
            return avg_mse, avg_nll, avg_mass, avg_total