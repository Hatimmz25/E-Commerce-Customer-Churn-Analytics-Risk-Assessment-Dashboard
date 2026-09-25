import streamlit as st
import pandas as pd
import numpy as np
from app.utils.predictor import get_risk_tier
from app.utils.model_loader import REQUIRED_FEATURES

def render_batch_prediction(model_pipeline, threshold):
    st.markdown('<h3 class="dashboard-header">📁 Batch Churn Prediction Engine</h3>', unsafe_allow_html=True)
    st.markdown("Upload a customer records CSV file to generate automated churn probabilities, predictions, and risk classifications.")

    uploaded_file = st.file_uploader("Upload Customer Batch CSV File", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file).copy()
        except Exception as e:
            st.error(f"❌ **Invalid CSV Format**: Unable to parse uploaded file. Error: {str(e)}")
            return

        st.success(f"File uploaded successfully. Total records detected: `{len(batch_df):,}`")

        # Standardize categorical column if present
        if 'PreferedOrderCat' in batch_df.columns:
            batch_df['PreferedOrderCat'] = batch_df['PreferedOrderCat'].replace({'Mobile': 'Mobile Phone'})

        uploaded_cols = batch_df.columns.tolist()
        missing_cols = [col for col in REQUIRED_FEATURES if col not in uploaded_cols]

        if missing_cols:
            st.error(f"❌ **Missing Required Columns**: Uploaded file is missing {len(missing_cols)} required input feature(s):")
            st.write(missing_cols)
            st.info(f"Required schema features: {REQUIRED_FEATURES}")
            return

        inference_df = batch_df[REQUIRED_FEATURES].copy()

        try:
            with st.spinner("Processing batch records through production model pipeline..."):
                batch_probs = model_pipeline.predict_proba(inference_df)[:, 1]

            batch_df['Churn_Probability'] = batch_probs
            batch_df['Predicted_Churn_Class'] = (batch_probs >= threshold).astype(int)
            batch_df['Risk_Level'] = batch_df['Churn_Probability'].apply(lambda p: get_risk_tier(p, threshold))

            st.markdown("---")
            st.markdown("##### Batch Execution Summary")

            b_kpi1, b_kpi2, b_kpi3, b_kpi4 = st.columns(4)
            high_risk_count = (batch_df['Risk_Level'] == 'HIGH RISK').sum()
            med_risk_count = (batch_df['Risk_Level'] == 'MEDIUM RISK').sum()
            low_risk_count = (batch_df['Risk_Level'] == 'LOW RISK').sum()

            b_kpi1.metric("Processed Batch Records", f"{len(batch_df):,}")
            b_kpi2.metric("High Churn Risk Customers", f"{high_risk_count:,}", f"{high_risk_count/len(batch_df):.1%}")
            b_kpi3.metric("Medium Risk Customers", f"{med_risk_count:,}", f"{med_risk_count/len(batch_df):.1%}")
            b_kpi4.metric("Low Risk / Retained", f"{low_risk_count:,}", f"{low_risk_count/len(batch_df):.1%}")

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("##### Batch Results Preview (First 10 Records)")
            
            preview_cols = [c for c in ['CustomerID', 'Tenure', 'PreferedOrderCat', 'DaySinceLastOrder', 'Churn_Probability', 'Predicted_Churn_Class', 'Risk_Level'] if c in batch_df.columns]
            st.dataframe(batch_df[preview_cols].head(10), use_container_width=True)

            st.markdown("<br>", unsafe_allow_html=True)
            csv_download = batch_df.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="📥 Download Complete Batch Predictions CSV",
                data=csv_download,
                file_name="ecommerce_churn_batch_predictions.csv",
                mime="text/csv",
                use_container_width=True
            )

        except Exception as b_err:
            st.error(f"❌ **Batch Processing Error**: An exception occurred during batch prediction. Details: {str(b_err)}")

    else:
        st.info("ℹ️ Upload a CSV file above to execute batch predictions.")
        st.markdown("##### Required CSV Schema Format Sample")
        sample_template = pd.DataFrame([{
            'Tenure': 12, 'WarehouseToHome': 15, 'NumberOfDeviceRegistered': 3,
            'PreferedOrderCat': 'Laptop & Accessory', 'SatisfactionScore': 4,
            'MaritalStatus': 'Single', 'NumberOfAddress': 2, 'Complain': 0,
            'DaySinceLastOrder': 5, 'CashbackAmount': 160.0
        }])
        st.dataframe(sample_template, use_container_width=True)