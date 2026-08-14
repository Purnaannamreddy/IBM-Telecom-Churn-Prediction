# Explainable Customer Churn Prediction & Model Robustness
### Sriram Annamreddy

---

## Project Overview

This project builds an end-to-end, explainable machine learning system to predict customer churn in the telecommunications industry. 

The primary goal is not just to build accurate predictive models, but also to:
1. Understand **why** customers leave using SHAP (Explainable AI).
2. Test how well the trained model **generalizes to a completely different telecom dataset** (Cell2Cell) without retraining.
3. Provide an easy-to-use **interactive Streamlit web dashboard** for business users to score individual customers and view their risk drivers.

---

## Project Structure & Workflow

The project is organized into 6 step-by-step Jupyter notebooks located in the `notebooks/` folder:

| Notebook | What It Does |
| :--- | :--- |
| **`01_data_preparation.ipynb`** | Loads raw IBM Telco data, handles missing values, creates simple domain features (e.g. Charges Per Tenure), sets up a leakage-free preprocessing pipeline, and performs an 80/20 stratified train/test split. |
| **`02_eda.ipynb`** | Explores churn distributions, contract types, monthly charges, and service subscriptions using charts and statistical tests (Chi-square and Point-biserial correlations). |
| **`03_model_training.ipynb`** | Trains and tunes three model architectures (**Logistic Regression, Random Forest, XGBoost**) across three class imbalance strategies (**None, Class Weight, SMOTE**) using 5-fold stratified cross-validation. |
| **`04_evaluation.ipynb`** | Evaluates all models on the held-out test set (ROC-AUC, PR-AUC, Brier score), checks probability calibration, and calculates a cost-sensitive decision threshold (assuming customer acquisition costs 5x more than retention). |
| **`05_shap_explainability.ipynb`** | Uses Tree/Linear SHAP to find global churn drivers (e.g. Month-to-month contracts, tenure, monthly charges), generates local explanations for high-risk customers, and groups churners into 3 behavioral segments using K-Means clustering on SHAP values. |
| **`06_external_validation_cell2cell.ipynb`** | Evaluates the frozen IBM Telco model on an external dataset (`cell2celltrain.csv`), runs a schema-mapping sensitivity analysis, and measures feature drift using the Population Stability Index (PSI). |

---

## Interactive Streamlit Web App (`app.py`)

A clean, single-page web dashboard is included to demonstrate real-world deployment:
- **Single Customer Scoring**: Input customer attributes (contract type, tenure, monthly charges, internet service) and get an immediate churn probability percentage and classification (`Stay` or `Churn`).
- **Live SHAP Explanations**: Automatically plots the top 5 features pushing that specific customer towards staying or churning.
- **Model Info Tab**: Shows model performance metrics and training churn baselines.

## How to Run the Streamlit Web Application
The interactive churn prediction dashboard is built with Streamlit in `app.py`.
### Step 1: Start the Streamlit Server
Open your terminal (PowerShell, Command Prompt, or Bash) in the project root directory and run:
```bash
streamlit run app.py
---



## Datasets Used

1. **IBM Telco Customer Churn (`IBM_Telco_customer_churn_IBM_dataset.csv`)**:
   - The primary dataset used for model training, validation, and evaluation (7,043 customers, 26.5% churn rate).
2. **Cell2Cell Dataset (`cell2celltrain.csv` & `cell2cellholdout.csv`)**:
   - An independent real-world mobile wireless dataset used strictly as an external test set to evaluate model robustness and measure data drift.

---

## How to Set Up and Run

### 1. Clone or Download the Repository
```bash
git clone https://github.com/Purnaannamreddy/telecom-churn-prediction-MSc-Project.git
cd telecom-churn-prediction-MSc-Project


