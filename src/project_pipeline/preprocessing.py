import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import numpy as np

def prepare_data(df):
    """
    Prepare dataset for training and testing:
    - Impute missing values
    - Split into train/test sets
    - Scale features
    - Balance training data with SMOTE

    Parameters:
    df (DataFrame): Input dataset containing features and 'Label' column

    Returns:
    X_train_bal (ndarray): Scaled and SMOTE-balanced training features
    X_test_scaled (ndarray): Scaled test features
    y_train_bal (ndarray): SMOTE-balanced training labels
    y_test (Series): Test labels
    scaler (StandardScaler): Fitted scaler (can be used for future data)
    """
    print("[DEBUG] Starting data preparation...")

    # Separate features and target
    X = df.drop(columns=["Label"])
    y = df["Label"].astype(float)
    print(f"[DEBUG] Features shape: {X.shape}, Target shape: {y.shape}")

    # Impute missing values using median
    imp = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imp.fit_transform(X), columns=X.columns, index=X.index)
    print("[DEBUG] Missing values imputed using median.")

    # Split into train and test sets (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"[DEBUG] Train/Test split: X_train {X_train.shape}, X_test {X_test.shape}")

    # Scale features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("[DEBUG] Features scaled with StandardScaler.")

    # Balance training data using SMOTE
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)
    print(f"[DEBUG] Before SMOTE: {np.bincount(y_train.astype(int))}")
    print(f"[DEBUG] After SMOTE: {np.bincount(y_train_bal.astype(int))}")

    return X_train_bal, X_test_scaled, y_train_bal, y_test, scaler

