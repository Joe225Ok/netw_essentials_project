# main_pipeline.py
import os
from project_pipeline.data_loader import load_cleaned_data
from project_pipeline.preprocessing import prepare_data
from project_pipeline.models_rf import train_rf
from project_pipeline.models_lr import train_lr
from project_pipeline.models_xgb import train_xgb
from project_pipeline.models_lgbm import train_lgb
from project_pipeline.models_cnn import train_cnn
from project_pipeline.utils_save_load import save_model, load_model
from project_pipeline.evaluation import evaluate_model
from project_pipeline.poster_materials import generate_poster_materials
from tensorflow.keras.models import load_model as load_keras_model

def main():
    # -----------------------------
    # Paths and Data
    # -----------------------------
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cleaned_data_path = os.path.join(current_dir, "../../data/Cleaned_Data_top_95_percent_features.csv")
    df = load_cleaned_data(cleaned_data_path)
    X_train_bal, X_test, y_train_bal, y_test, _ = prepare_data(df)

    models_dir = os.path.join(current_dir, "../../data/models")
    os.makedirs(models_dir, exist_ok=True)

    # -----------------------------
    # Model Save Paths
    # -----------------------------
    model_paths = {
        "rf": os.path.join(models_dir, "rf_model.pkl"),
        "lr": os.path.join(models_dir, "lr_model.pkl"),
        "xgb": os.path.join(models_dir, "xgb_model.pkl"),
        "lgb": os.path.join(models_dir, "lgb_model.pkl"),
        "cnn": os.path.join(models_dir, "cnn_model.h5")
    }

    metrics_dict = {}  # To store Accuracy, F1, ROC-AUC for all models

    # -----------------------------
    # Random Forest
    # -----------------------------
    if os.path.exists(model_paths["rf"]):
        rf = load_model(model_paths["rf"])
        print("Loaded saved Random Forest model")
    else:
        rf = train_rf(X_train_bal, y_train_bal)
        save_model(rf, model_paths["rf"])
        print("Random Forest trained and saved")

    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]
    metrics_dict["Random Forest"] = evaluate_model("Random Forest", y_test, y_pred, y_prob)

    # -----------------------------
    # Logistic Regression
    # -----------------------------
    if os.path.exists(model_paths["lr"]):
        lr = load_model(model_paths["lr"])
        print("Loaded saved Logistic Regression model")
    else:
        lr = train_lr(X_train_bal, y_train_bal)
        save_model(lr, model_paths["lr"])
        print("Logistic Regression trained and saved")

    y_pred = lr.predict(X_test)
    y_prob = lr.predict_proba(X_test)[:, 1]
    metrics_dict["Logistic Regression"] = evaluate_model("Logistic Regression", y_test, y_pred, y_prob)

    # -----------------------------
    # XGBoost
    # -----------------------------
    if os.path.exists(model_paths["xgb"]):
        xgb = load_model(model_paths["xgb"])
        print("Loaded saved XGBoost model")
    else:
        xgb = train_xgb(X_train_bal, y_train_bal)
        save_model(xgb, model_paths["xgb"])
        print("XGBoost trained and saved")

    y_pred = xgb.predict(X_test)
    y_prob = xgb.predict_proba(X_test)[:, 1]
    metrics_dict["XGBoost"] = evaluate_model("XGBoost", y_test, y_pred, y_prob)

    # -----------------------------
    # LightGBM
    # -----------------------------
    if os.path.exists(model_paths["lgb"]):
        lgb = load_model(model_paths["lgb"])
        print("Loaded saved LightGBM model")
    else:
        lgb = train_lgb(X_train_bal, y_train_bal)
        save_model(lgb, model_paths["lgb"])
        print("LightGBM trained and saved")

    y_pred = lgb.predict(X_test)
    y_prob = lgb.predict_proba(X_test)[:, 1]
    metrics_dict["LightGBM"] = evaluate_model("LightGBM", y_test, y_pred, y_prob)

    # -----------------------------
    # CNN
    # -----------------------------
    if os.path.exists(model_paths["cnn"]):
        cnn = load_keras_model(model_paths["cnn"])
        print("Loaded saved CNN model")
    else:
        cnn = train_cnn(X_train_bal, y_train_bal)
        cnn.save(model_paths["cnn"])
        print("CNN trained and saved")

    y_prob = cnn.predict(X_test).flatten()           
    y_pred = (y_prob > 0.5).astype(int)           

    metrics_dict["CNN"] = evaluate_model("CNN", y_test, y_pred, y_prob)

    # -----------------------------
    # Generate Poster Materials
    # -----------------------------
    generate_poster_materials(
        df=df,
        metrics_dict=metrics_dict,
        output_dir=os.path.join(current_dir, "../../data/poster_materials")
    )

if __name__ == "__main__":
    main()
