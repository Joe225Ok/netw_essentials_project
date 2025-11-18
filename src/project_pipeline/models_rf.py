# Enable HalvingGridSearchCV (experimental module)
from sklearn.experimental import enable_halving_search_cv  # noqa

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import HalvingGridSearchCV, KFold
import time

def train_rf(X, y, seed=42):
    """
    Train a Random Forest classifier using HalvingGridSearchCV (much faster) 
    while preserving the search logic and providing progress updates.

    Parameters:
    X : array-like
        Features for training
    y : array-like
        Target labels
    seed : int
        Random seed for reproducibility

    Returns:
    best_estimator_ : RandomForestClassifier
        Best estimator found by HalvingGridSearchCV
    """
    print("\n[DEBUG] Initializing optimized Random Forest training...")
    start_time = time.time()

    # Reduced but meaningful search space (faster)
    params = {
        'n_estimators': [200, 400],         # fewer values → faster
        'max_features': ['sqrt', 0.3],
        'min_samples_leaf': [1, 2],
        'min_samples_split': [2, 4],
        'class_weight': ['balanced']
    }

    # Deterministic cross-validation
    # cv = KFold(n_splits=3, shuffle=True, random_state=seed)

    print("[DEBUG] Parameter grid defined.")
    print("[DEBUG] Using HalvingGridSearchCV for fast hyperparameter search...")

    grid = HalvingGridSearchCV(
        RandomForestClassifier(random_state=seed, n_jobs=-1),
        params,
        scoring='f1_macro',
        cv=3,
        factor=2,
        verbose=1,
        n_jobs=-1,
        aggressive_elimination=True # further speeds up the search
    )

    print("\n[DEBUG] Starting model search...")
    grid.fit(X, y)

    elapsed = time.time() - start_time
    print(f"\n[DEBUG] Random Forest search completed in {elapsed:.2f} seconds")
    print(f"[DEBUG] Best Params: {grid.best_params_}")
    print(f"[DEBUG] Best F1-macro Score: {grid.best_score_:.4f}")

    return grid.best_estimator_
