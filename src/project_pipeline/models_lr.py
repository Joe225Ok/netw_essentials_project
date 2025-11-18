from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import time
import numpy as np
import random

# Global seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

def train_lr(X, y, seed=SEED):
    """
    Train a Logistic Regression classifier using GridSearchCV with timing and reproducibility.

    Parameters:
    X : array-like
        Features for training
    y : array-like
        Target labels
    seed : int
        Random seed for reproducibility

    Returns:
    best_model : LogisticRegression
        Best estimator found by GridSearchCV
    """
    print("\n[DEBUG] Setting up hyperparameter grid for Logistic Regression...")
    start_time = time.time()

    params = {
        'C': [0.1, 1, 3, 5],          # Inverse of regularization strength
        'penalty': ['l2'],            # Regularization type
        'solver': ['lbfgs'],          # Optimization algorithm
        'class_weight': ['balanced']  # Handle class imbalance
    }

    print("[DEBUG] Initializing GridSearchCV for Logistic Regression...")
    grid = GridSearchCV(
        LogisticRegression(max_iter=2000, random_state=seed),
        params,
        scoring='f1_macro',
        cv=3,
        n_jobs=-1,
        verbose=1
    )

    print("[DEBUG] Fitting Logistic Regression GridSearchCV...")
    grid.fit(X, y)

    elapsed = time.time() - start_time
    print(f"[DEBUG] Logistic Regression search completed in {elapsed:.2f} seconds")
    print(f"[DEBUG] Best params: {grid.best_params_}")
    print(f"[DEBUG] Best F1-macro score: {grid.best_score_:.4f}")

    return grid.best_estimator_
