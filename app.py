import pathlib
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import joblib
import streamlit as st

# Setup project directories
PROJECT_ROOT = Path.cwd()
if (PROJECT_ROOT / "models").exists():
    MODEL_DIR = PROJECT_ROOT / "models"
    DATA_DIR = PROJECT_ROOT / "data" / "processed"
    TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
else:
    MODEL_DIR = PROJECT_ROOT.parent / "models"
    DATA_DIR = PROJECT_ROOT.parent / "data" / "processed"
    TABLES_DIR = PROJECT_ROOT.parent / "outputs" / "tables"

@st.cache_resource
def load_artifacts():
    bundle = joblib.load(MODEL_DIR / "best_model.joblib")
    preprocessor = joblib.load(MODEL_DIR / "preprocessor.joblib")
    cleaned_data = pd.read_csv(DATA_DIR / "cleaned_data.csv")
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    
    test_metrics_path = TABLES_DIR / "test_metrics.csv"
    if not test_metrics_path.exists():
        test_metrics_path = DATA_DIR / "test_metrics.csv"
    test_metrics = pd.read_csv(test_metrics_path) if test_metrics_path.exists() else None
    
    return bundle, preprocessor, cleaned_data, X_train, test_metrics

def main():
    st.set_page_config(page_title="Telco Churn Prediction", layout="wide")
    
    bundle, preprocessor, cleaned_data, X_train, test_metrics = load_artifacts()
    
    pipeline = bundle["model"]
    threshold = float(bundle["threshold"])
    best_name = bundle["name"]
    
    st.title("Telco Churn Prediction")
    
    tab_predict, tab_info = st.tabs(["Single Prediction", "Model Info"])
    
    with tab_predict:
        st.write("The model was trained on the IBM Telco Customer Churn dataset; predictions represent estimated probabilities, not guaranteed individual outcomes.")
        st.header("New customer")
        
        # Derive ranges and default options dynamically from cleaned_data.csv
        min_tenure = int(cleaned_data["Tenure Months"].min())
        max_tenure = int(cleaned_data["Tenure Months"].max())
        med_tenure = int(cleaned_data["Tenure Months"].median())
        
        min_monthly = float(cleaned_data["Monthly Charges"].min())
        max_monthly = float(cleaned_data["Monthly Charges"].max())
        med_monthly = float(cleaned_data["Monthly Charges"].median())
        
        contract_opts = sorted(cleaned_data["Contract"].dropna().unique().tolist())
        internet_opts = sorted(cleaned_data["Internet Service"].dropna().unique().tolist())
        payment_opts = sorted(cleaned_data["Payment Method"].dropna().unique().tolist())
        
        with st.form("single_customer_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Account & Demographics")
                tenure_months = st.slider("Tenure Months", min_value=min_tenure, max_value=max_tenure, value=med_tenure)
                monthly_charges = st.number_input("Monthly Charges", min_value=min_monthly, max_value=max_monthly, value=med_monthly, step=0.5)
                gender = st.selectbox("Gender", ["Male", "Female"])
                senior_citizen = st.selectbox("Senior Citizen", [0, 1])
                partner = st.selectbox("Partner", ["Yes", "No"])
                dependents = st.selectbox("Dependents", ["Yes", "No"])
                
            with col2:
                st.subheader("Contract & Billing")
                contract = st.selectbox("Contract", contract_opts)
                internet_service = st.selectbox("Internet Service", internet_opts)
                payment_method = st.selectbox("Payment Method", payment_opts)
                phone_service = st.selectbox("Phone Service", ["Yes", "No"])
                multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No"])
                
            with col3:
                st.subheader("Add-on Services")
                online_security = st.selectbox("Online Security", ["Yes", "No"])
                online_backup = st.selectbox("Online Backup", ["Yes", "No"])
                device_protection = st.selectbox("Device Protection", ["Yes", "No"])
                tech_support = st.selectbox("Tech Support", ["Yes", "No"])
                streaming_tv = st.selectbox("Streaming TV", ["Yes", "No"])
                streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No"])
                
            submitted = st.form_submit_button("Predict")
            
        if submitted:
            # Recompute engineered features inside the app matching 01_data_preparation.ipynb
            paperless_billing = "Yes"
            total_charges = float(monthly_charges * tenure_months)
            avg_monthly = total_charges / (tenure_months if tenure_months > 0 else 1)
            charges_per_tenure = monthly_charges / (tenure_months + 1)
            tenure_band = pd.cut([tenure_months], bins=[-1, 12, 24, 48, np.inf], labels=["0-12", "13-24", "25-48", "49+"])[0]
            high_monthly = int(monthly_charges >= 70)
            long_tenure = int(tenure_months > 24)
            
            service_vals = [online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies]
            multiple_services = sum(1 for v in service_vals if v == "Yes")
            has_phone = int(phone_service == "Yes")
            has_internet = int(internet_service != "No")
            
            input_dict = {
                "Gender": gender,
                "Senior Citizen": senior_citizen,
                "Partner": partner,
                "Dependents": dependents,
                "Tenure Months": tenure_months,
                "Phone Service": phone_service,
                "Multiple Lines": multiple_lines,
                "Internet Service": internet_service,
                "Online Security": online_security,
                "Online Backup": online_backup,
                "Device Protection": device_protection,
                "Tech Support": tech_support,
                "Streaming TV": streaming_tv,
                "Streaming Movies": streaming_movies,
                "Contract": contract,
                "Paperless Billing": paperless_billing,
                "Payment Method": payment_method,
                "Monthly Charges": monthly_charges,
                "Total Charges": total_charges,
                "Avg Monthly Charges": avg_monthly,
                "Charges Per Tenure": charges_per_tenure,
                "Tenure Band": tenure_band,
                "High Monthly Charges": high_monthly,
                "Long Tenure": long_tenure,
                "Multiple Services": multiple_services,
                "Has Phone": has_phone,
                "Has Internet": has_internet
            }
            
            input_df = pd.DataFrame([input_dict])
            
            pred_prob = float(pipeline.predict_proba(input_df)[0, 1])
            is_churn = pred_prob >= threshold
            
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric("Predicted churn probability", f"{pred_prob:.1%}")
            with col_m2:
                st.metric("Decision threshold", f"{threshold:.1%}")
                
            if is_churn:
                st.markdown("### Prediction: Churn")
            else:
                st.markdown("### Prediction: Stay")
                
            st.write("Higher risk indicates a greater estimated probability of churn; this output supports review and is not a guaranteed outcome.")
            
            st.subheader("Top risk drivers")
            
            prep = pipeline.named_steps["preprocess"]
            estimator = pipeline.named_steps["model"]
            
            transformed_input = prep.transform(input_df)
            transformed_train = prep.transform(X_train)
            
            feature_names = [col.replace("num__", "").replace("cat__", "") for col in prep.get_feature_names_out()]
            
            explainer = shap.Explainer(estimator, transformed_train, feature_names=feature_names)
            shap_values = explainer(transformed_input)
            
            fig, ax = plt.subplots(figsize=(8, 3))
            shap.plots.bar(shap_values[0], max_display=5, show=False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    with tab_info:
        st.header("Model Info")
        
        churn_rate = float(cleaned_data["Churn Value"].mean())
        
        st.write(f"**Best Model Candidate:** {best_name}")
        st.write(f"**Training Set Churn Rate:** {churn_rate:.2%}")
        
        if test_metrics is not None:
            best_row = test_metrics[test_metrics["candidate"] == best_name]
            if not best_row.empty:
                best_auc = float(best_row["roc_auc"].values[0])
                st.write(f"**IBM Telco Test Set ROC-AUC:** {best_auc:.4f}")
            
            st.subheader("Model Comparison Summary")
            st.dataframe(test_metrics, use_container_width=True)

if __name__ == "__main__":
    main()
