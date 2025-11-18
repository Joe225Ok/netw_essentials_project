import os
import pandas as pd

def generate_poster_materials(df, metrics_dict=None, output_dir="../../data/poster_materials"):
    """
    Generate text summary and a performance table for a cybersecurity scientific poster.
    
    Parameters:
        df (pd.DataFrame): Cleaned dataset (after feature selection)
        metrics_dict (dict): Dictionary with evaluation metrics for all models
        output_dir (str): Directory to save poster materials
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Poster materials will be saved to: {output_dir}")

    # -----------------------------
    # 1. Text Summary
    # -----------------------------
    summary_text = f"""
==============================
PROJECT IMPLEMENTATION SUMMARY
==============================

● Dataset Size: {df.shape[0]:,} rows
● Total Initial Features: 78
● Features After Cleaning + LGBM Feature Selection: 35 
● 35 features contribute to 95% of overall model importance.

----------------------------
ORIGINAL LABEL CLASSES (13)
----------------------------
Benign
DoS GoldenEye
Heartbleed
DoS Hulk
DoS SlowHTTP
DoS Slowloris
SSH-Patator
FTP-Patator
Web Attack
Infiltration
Bot
PortScan
DDoS

----------------------------
TRANSFORMED LABEL STRUCTURE
----------------------------
● BENIGN  → TRUSTED
● All other attack types → UNTRUSTED
  (binary classification)

----------------------------
MACHINE LEARNING PIPELINE
----------------------------
1. Load original dataset (2 million+ rows)
2. Clean data, handle missing values, normalize
3. Apply LGBM feature importance → keep top 35 contributors
4. Encode TRUSTED/UNTRUSTED target
5. Split into train/test sets
6. Train ML models: RF / LR / XGB / LGBM / CNN
7. Save trained models to /data/models/
8. Evaluate and generate performance metrics
"""
    summary_file = os.path.join(output_dir, "poster_text_summary.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(summary_text)
    print("[INFO] Poster text summary saved.")

    # -----------------------------
    # 2. Model Performance Table
    # -----------------------------
    if metrics_dict:
        metrics_df = pd.DataFrame(metrics_dict).T
        metrics_df = metrics_df.rename(columns={0: "Accuracy", 1: "F1-macro", 2: "ROC-AUC"})
        metrics_df.to_csv(os.path.join(output_dir, "model_performance_table.csv"))
        print("[INFO] Model performance table saved.")

    print("\n[✔] Poster materials generated successfully!\n")
