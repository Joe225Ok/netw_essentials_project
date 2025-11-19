# Network Essentials Project

This repository contains a complete machine learning pipeline for preprocessing, feature engineering, training, and evaluating models using the **CIC-IDS-2017 Intrusion Detection Dataset**. It includes data wrangling notebooks, model training scripts, evaluation utilities, and feature importance extraction. Several large files are excluded due to GitHub’s 100 MB file-size limit.

---

## Repository Structure

- **data/**  
  Contains dataset files. Large files are excluded (see list below).
- **data/models/**  
  Contains trained models. All models are excluded due to size.
- **notebook/data_wrangling.ipynb**  
  Jupyter Notebook used to preprocess the original CIC-IDS-2017 dataset and generate cleaned feature sets.
- **src/main_pipeline.py**  
  Main machine learning pipeline that loads data, trains models, evaluates performance, and saves outputs.

---

## Important: Large Files Excluded From Repository

The following files were removed from version control because they exceed GitHub’s 100 MB limit:

### Excluded Model Files
- data/models/cnn_model.h5  
- data/models/lgb_model.pkl  
- data/models/lr_model.pkl  
- data/models/rf_model.pkl  
- data/models/xgb_model.pkl  

### Excluded Data Files
- data/Cleaned_Data_top_95_percent_features.csv  
- data/FullData.csv  
- data/GeneratedLabelledFlows.zip  
- data/MachineLearningCSV.zip  
- data/top_95_percent_features.csv  

These files must be manually restored before running the full pipeline.

---

## How to Restore Missing Data

You may restore the excluded files in **one of two ways**.

### Option 1 — Download CIC-IDS-2017 Yourself (Recommended)

Download the dataset from the **Canadian Institute for Cybersecurity**:

**https://www.unb.ca/cic/datasets/ids-2017.html**

Then:

1. Place **MachineLearningCSV.zip** into the `data/` directory.  
2. Open and run `notebook/data_wrangling.ipynb`.  
   This notebook will generate:
   - Cleaned_Data_top_95_percent_features.csv  
   - top_95_percent_features.csv  

This ensures the project can run from scratch.

### Option 2 — Request the Processed Data and Models

You may request the cleaned CSV files and trained model files.  
After receiving them:

- Place all CSV files into the `data/` folder  
- Place all model files into `data/models/`  

---

## Running the Project

After restoring the dataset:

1. Run *data_wrangling.ipynb* to generate the cleaned features (if not provided).  
2. Execute *src/main_pipeline.py* to train or load models and perform evaluation.

The pipeline will automatically use or generate the following model files:

- data/models/cnn_model.h5  
- data/models/lgb_model.pkl  
- data/models/lr_model.pkl  
- data/models/rf_model.pkl  
- data/models/xgb_model.pkl  

---

## About the Dataset (CIC-IDS-2017)

This project is built on the **CIC-IDS-2017 dataset**, created by the  
**Canadian Institute for Cybersecurity (CIC)** at the **University of New Brunswick (UNB)**.  
It provides realistic network traffic including benign and malicious flows, commonly used in intrusion detection research.

Dataset homepage:  
**https://www.unb.ca/cic/datasets/ids-2017.html**

---


