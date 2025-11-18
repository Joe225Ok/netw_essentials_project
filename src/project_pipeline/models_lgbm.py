# models_lgbm.py
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV
import time
import numpy as np
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

def train_lgb(X, y, seed=SEED):
    """
    Train a LightGBM classifier using RandomizedSearchCV
    with timing, reproducibility, and reduced training time.
    """
    print("\n[DEBUG] Setting up hyperparameter search space for LightGBM...")
    start_time = time.time()

    # Hyperparameter search space
    params = {
        'n_estimators': [200, 300, 400, 600],
        'max_depth': [-1, 6, 8, 12],
        'learning_rate': np.linspace(0.01, 0.2, 5),
        'num_leaves': [31, 63, 127, 255],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }

    print("[DEBUG] Initializing RandomizedSearchCV for LightGBM...")

    search = RandomizedSearchCV(
        estimator=LGBMClassifier(random_state=seed),
        param_distributions=params,
        n_iter=25,                     # Only 25 random combos
        scoring='f1_macro',
        cv=3,
        n_jobs=-1,
        random_state=seed,
        verbose=1
    )

    print("[DEBUG] Fitting LightGBM RandomizedSearchCV...")
    search.fit(X, y)

    elapsed = time.time() - start_time
    print(f"[DEBUG] LightGBM Randomized Search completed in {elapsed:.2f} seconds")
    print(f"[DEBUG] Best params: {search.best_params_}")
    print(f"[DEBUG] Best F1-macro: {search.best_score_:.4f}")

    return search.best_estimator_
