import os
import json
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)


def evaluate_model(name, y_true, y_pred, y_prob, results_dir="results"):
    """
    Evaluate a classification model, print and save metrics.
    """
    os.makedirs(results_dir, exist_ok=True)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    auc = roc_auc_score(y_true, y_prob)

    print(f"\n===== {name} Evaluation =====")
    print(f"Accuracy: {acc:.4f}, F1-macro: {f1:.4f}, AUC: {auc:.4f}")
    print(classification_report(y_true, y_pred))

    # Save metrics JSON
    metrics = {"model": name, "accuracy": acc, "f1_macro": f1, "auc": auc}
    with open(f"{results_dir}/{name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # Save classification report CSV
    report_df = pd.DataFrame(
        classification_report(y_true, y_pred, output_dict=True)
    ).transpose()
    report_df.to_csv(f"{results_dir}/{name}_classification_report.csv")

    return metrics


def plot_confusion(name, y_true, y_pred, results_dir="results"):
    """
    Generate and save confusion matrix.
    """
    os.makedirs(results_dir, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.savefig(f"{results_dir}/{name}_confusion_matrix.png", dpi=300)
    plt.close()
