import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
import pickle
import os

def data_import(random_state=42, noise_std=0.01, csv_file="CMEHL_physics.csv", use_log_transform=True, epsilon=1e-10):
    """
    Import data and create input-output pairs for time series prediction with scaling.
    
    Args:
        random_state: Random seed for reproducibility
        noise_std: Standard deviation of the noise added to training data
        csv_file: Path to the CSV file containing the data
        use_log_transform: Whether to apply log transformation to mass fractions
        epsilon: Small number added to mass fractions to avoid log(0)
        
    Returns:
        X_train, X_val, Y_train, Y_val, X_test, Y_test, scaler, transform_info
    """
    # Read the data
    df = pd.read_csv(csv_file)
    
    # Separate by dataset type
    train_df = df[df['dataset_type'] == 'train']
    test_df = df[df['dataset_type'] == 'test']
    val_df = df[df['dataset_type'] == 'validation']
    
    # Get feature columns (all except dataset_type)
    features = df.columns[:-1].tolist()
    
    # Store transformation information
    transform_info = {
        'use_log_transform': use_log_transform,
        'epsilon': epsilon,
        'temp_index': 0,  # Index of temperature column
        'species_indices': list(range(1, 10))  # Indices of mass fraction columns
    }
    
    # Apply log transformation if requested
    if use_log_transform:
        # We'll apply log transform only to mass fractions, not to temperature
        for dataset in [train_df, test_df, val_df]:
            # Temperature doesn't need log transform
            temp_values = dataset[features[0]].values
            
            # Apply log to mass fractions (add epsilon to avoid log(0))
            for i in range(1, len(features)):
                dataset[features[i]] = np.log(dataset[features[i]] + epsilon)
    
    # Initialize scaler
    scaler = StandardScaler()
    
    # Fit scaler on training data only to avoid data leakage
    scaler.fit(train_df[features])
    
    # Transform all datasets
    train_scaled = scaler.transform(train_df[features])
    test_scaled = scaler.transform(test_df[features])
    val_scaled = scaler.transform(val_df[features])
    
    # Create input-output pairs for each set
    X_train = train_scaled[:-1]
    Y_train = train_scaled[1:]
    
    X_test = test_scaled[:-1]
    Y_test = test_scaled[1:]
    
    X_val = val_scaled[:-1]
    Y_val = val_scaled[1:]
    
    # Add noise to training data
    np.random.seed(random_state)
    Y_train += np.random.normal(0, noise_std, Y_train.shape)
    
    return X_train, X_val, Y_train, Y_val, X_test, Y_test, scaler, transform_info

def save_transform_info(transform_info, directory='models', filename='transform_info.pkl'):
    """Save transformation information to a file"""
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, filename), 'wb') as f:
        pickle.dump(transform_info, f)

def load_transform_info(directory='models', filename='transform_info.pkl'):
    """Load transformation information from a file"""
    path = os.path.join(directory, filename)
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

def inverse_transform_predictions(predictions, scaler, transform_info):
    """
    Convert predictions back to the original scale
    
    Args:
        predictions: Model predictions in the scaled/transformed space
        scaler: StandardScaler used for scaling
        transform_info: Dictionary with transformation information
        
    Returns:
        predictions_original: Predictions in the original scale
    """
    # First undo the scaling
    predictions_unscaled = scaler.inverse_transform(predictions)
    
    # If log transform wasn't used, return the unscaled predictions
    if not transform_info['use_log_transform']:
        return predictions_unscaled
    
    # If log transform was used, need to exponentiate the mass fractions
    predictions_original = predictions_unscaled.copy()
    
    # Exponentiate mass fractions (and subtract epsilon that was added before log)
    for idx in transform_info['species_indices']:
        predictions_original[:, idx] = np.exp(predictions_unscaled[:, idx]) - transform_info['epsilon']
        # Ensure no negative values
        predictions_original[:, idx] = np.maximum(0, predictions_original[:, idx])
        
    return predictions_original