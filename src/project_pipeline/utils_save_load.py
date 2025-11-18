import joblib

def save_model(model, path):
    """
    Save a trained model to disk using joblib.

    Parameters:
    model: Trained model object (e.g., scikit-learn model)
    path (str): File path where the model will be saved
    """
    print(f"[DEBUG] Saving model to: {path}")
    joblib.dump(model, path)
    print("[DEBUG] Model saved successfully.")

def load_model(path):
    """
    Load a trained model from disk using joblib.

    Parameters:
    path (str): File path of the saved model

    Returns:
    model: Loaded model object
    """
    print(f"[DEBUG] Loading model from: {path}")
    model = joblib.load(path)
    print("[DEBUG] Model loaded successfully.")
    return model
