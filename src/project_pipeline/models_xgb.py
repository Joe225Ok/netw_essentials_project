# models_xgb.py
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import time
import numpy as np
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

def train_xgb(X, y, seed=SEED):
    """
    Train an XGBoost classifier using GridSearchCV with timing and reproducibility.
    """
    print("\n[DEBUG] Setting up hyperparameter grid for XGBoost...")
    start_time = time.time()

    params = {
        'n_estimators': [200, 400],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 1],
        'colsample_bytree': [0.7, 1],
        'scale_pos_weight': [1, 3]
    }

    print("[DEBUG] Initializing GridSearchCV for XGBoost...")
    grid = GridSearchCV(
        XGBClassifier(random_state=seed, eval_metric='logloss', use_label_encoder=False),
        params,
        scoring='f1_macro',
        cv=3,
        n_jobs=-1,
        verbose=1
    )

    print("[DEBUG] Fitting XGBoost GridSearchCV...")
    grid.fit(X, y)

    elapsed = time.time() - start_time
    print(f"[DEBUG] XGBoost search completed in {elapsed:.2f} seconds")
    print(f"[DEBUG] Best params: {grid.best_params_}")
    print(f"[DEBUG] Best F1-macro score: {grid.best_score_:.4f}")

    return grid.best_estimator_
