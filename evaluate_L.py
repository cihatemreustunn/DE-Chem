import numpy as np

def evaluate_model(model, X_heldout, Y_heldout, scaler):
    mean, variance = model.predict(X_heldout)
    
    # Inverse transform predictions and true values
    mean_unscaled = scaler.inverse_transform(mean)
    Y_unscaled = scaler.inverse_transform(Y_heldout)
    
    # Transform variance to unscaled space
    variance_unscaled = variance * (scaler.scale_**2)
    
    mse = np.mean((Y_unscaled - mean_unscaled)**2)
    uncertainty = np.mean(np.sqrt(variance_unscaled))
    return mse, uncertainty, mean_unscaled, variance_unscaled