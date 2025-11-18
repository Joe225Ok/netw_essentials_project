import time
import numpy as np
import random
import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Set global seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

def build_cnn(hp):
    """
    Build a Keras Sequential CNN model with tunable hyperparameters.
    """
    print("[DEBUG] Building CNN model with hyperparameters...")
    model = Sequential([
        Dense(
            hp.Int('units1', 64, 256, 64),
            activation='relu',
            input_dim=35
        ),
        Dropout(hp.Float('dropout1', 0.2, 0.5, 0.1)),
        Dense(
            hp.Int('units2', 32, 128, 32),
            activation='relu'
        ),
        Dropout(hp.Float('dropout2', 0.2, 0.5, 0.1)),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    print("[DEBUG] CNN model compiled.")
    return model

def train_cnn(X, y, seed=SEED):
    """
    Train a CNN model using Keras Tuner RandomSearch with timing and reproducibility.
    """
    print("[DEBUG] Initializing Keras Tuner RandomSearch...")
    start_time = time.time()

    tuner = kt.RandomSearch(
        build_cnn,
        objective='val_accuracy',
        max_trials=8,
        directory='cnn_tuner',
        project_name='project',
        seed=seed
    )

    es = EarlyStopping(
        monitor='val_loss',
        patience=4,
        restore_best_weights=True
    )

    print("[DEBUG] Starting hyperparameter search...")
    tuner.search(
        X, y,
        validation_split=0.2,
        epochs=40,
        batch_size=256,
        callbacks=[es]
    )

    elapsed = time.time() - start_time
    print(f"[DEBUG] Hyperparameter search completed in {elapsed:.2f} seconds")

    best_model = tuner.get_best_models(1)[0]
    print("[DEBUG] Best model retrieved.")

    return best_model
