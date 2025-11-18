import shap
from sklearn.inspection import permutation_importance
import numpy as np

def shap_importance(model, X):
    """
    Compute SHAP values for a given model and dataset.
    
    Parameters:
    model: Trained model (tree-based models are supported by TreeExplainer)
    X (array-like or DataFrame): Input features for explanation
    
    Returns:
    shap_values (array-like): SHAP values for each feature
    """
    print("[DEBUG] Initializing SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    
    print("[DEBUG] Calculating SHAP values...")
    shap_values = explainer.shap_values(X)
    print(f"[DEBUG] SHAP values shape: {np.array(shap_values).shape}")
    
    return shap_values

def permutation_imp(model, X, y):
    """
    Compute permutation feature importance.
    
    Parameters:
    model: Trained model
    X (array-like or DataFrame): Features
    y (array-like): True target values
    
    Returns:
    importances_mean (array): Mean importance for each feature
    """
    print("[DEBUG] Calculating permutation feature importance...")
    result = permutation_importance(model, X, y, n_repeats=10, n_jobs=-1)
    print(f"[DEBUG] Permutation importances mean: {result.importances_mean}")
    
    return result.importances_mean

def combine_importance(shap_vals, perm_vals):
    """
    Combine SHAP and permutation importances into a single importance score.
    
    Parameters:
    shap_vals (array-like): SHAP values for features
    perm_vals (array-like): Permutation importances for features
    
    Returns:
    combined (array): Combined importance values
    """
    print("[DEBUG] Normalizing SHAP values and combining with permutation importances...")
    shap_norm = np.abs(shap_vals).mean(axis=0)  # Take mean absolute SHAP value per feature
    perm_norm = perm_vals  # Already mean values from permutation importance
    combined = shap_norm + perm_norm
    print(f"[DEBUG] Combined importance calculated: {combined}")
    
    return combined

