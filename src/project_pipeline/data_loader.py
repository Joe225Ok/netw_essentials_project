import os
import pandas as pd

def load_cleaned_data(path: str):
    """
    Load a cleaned CSV dataset from a given file path.

    Parameters:
    path (str): File path to the CSV file

    Returns:
    df (DataFrame): Loaded Pandas DataFrame
    """
    print(f"[DEBUG] Checking if file exists at: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    print(f"[DEBUG] Loading CSV file from: {path}")
    df = pd.read_csv(path)
    print(f"[DEBUG] Loaded DataFrame with {len(df)} rows × {len(df.columns)} columns")
    
    return df
