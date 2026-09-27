import streamlit as st
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import shap
from app.utils.predictor import predict_single

def render_single_prediction(model_pipeline, threshold, metadata):
    with st.form(key='analytics_input_form'):
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.markdown('<div class="section-title">1. Customer Demographics</div>', unsafe_allow_html=True)
            tenure = st.number_input("Account Tenure (Months)", min_value=0, max_value=120, value=12, step=1)
            marital_status = st.selectbox("Marital Status", options=["Single", "Married", "Divorced"])
            num_addresses = st.number_input("Saved Shipping Addresses", min_value=1, max_value=25, value=2, step=1)

        with col_b:
            st.markdown('<div class="section-title">2. Behavioral Activity</div>', unsafe_allow_html=True)
            days_since_last_order = st.number_input("Days Since Last Purchase", min_value=0, max_value=365, value=7, step=1)
            num_devices = st.number_input("Registered Devices", min_value=1, max_value=10, value=3, step=1)
            satisfaction_score = st.slider("Satisfaction Score (1 - 5)", min_value=1, max_value=5, value=3, step=1)
            complain = st.radio("Logged Complaint?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", horizontal=True)

        with col_c:
            st.markdown('<div class="section-title">3. Financial & Logistics</div>', unsafe_allow_html=True)
            preferred_cat = st.selectbox("Primary Order Category", options=["Laptop & Accessory", "Mobile Phone", "Fashion", "Grocery", "Others"])
            cashback = st.number_input("Average Cashback Earned ($)", min_value=0.0, max_value=1000.0, value=150.0, step=10.0)
            warehouse_dist = st.number_input("Fulfillment Distance (km)", min_value=0, max_value=200, value=15, step=1)

        st.markdown("<br>", unsafe_allow_html=True)
        submit_button = st.form_submit_button(label="Evaluate Customer Risk Profile", use_container_width=True)

    if submit_button:
        # Standardize category inputs matching training mapping
        cat_value = 'Mobile Phone' if preferred_cat in ['Mobile', 'Mobile Phone'] else preferred_cat

        # Construct Raw Inputs DataFrame (Matching Pipeline Schema Exactly)
        raw_input_df = pd.DataFrame([{
            'Tenure': tenure,
            'WarehouseToHome': warehouse_dist,
            'NumberOfDeviceRegistered': num_devices,
            'PreferedOrderCat': cat_value,
            'SatisfactionScore': satisfaction_score,
            'MaritalStatus': marital_status,
            'NumberOfAddress': num_addresses,
            'Complain': complain,
            'DaySinceLastOrder': days_since_last_order,
            'CashbackAmount': cashback
        }])

        # Predict using full pipeline
        res = predict_single(model_pipeline, raw_input_df, threshold)

        st.markdown("---")
        st.markdown('<h3 class="dashboard-header">🎯 Prediction Dashboard</h3>', unsafe_allow_html=True)

        card1, card2, card3, card4 = st.columns(4)
        card1.metric("Probability of Churn", f"{res['probability']:.1%}")
        card2.metric("Predicted Churn Class", "Class 1 (Churn)" if res['predicted_class'] == 1 else "Class 0 (Retained)")
        card3.metric("Risk Classification", res['risk_level'])
        card4.metric("Classification Threshold", f"{threshold:.1%}")

        st.markdown("<br>", unsafe_allow_html=True)
        prob_col, banner_col = st.columns([1, 1])

        with prob_col:
            st.markdown("##### 📈 Visual Probability Meter")
            st.progress(res['probability'])
            st.caption(f"**0.0%** (Low Risk) ⟵ **{threshold:.1%} Cutoff** ⟶ **100.0%** (High Risk)")

        with banner_col:
            st.markdown(f"""
            <div class="{res['risk_style']}">
                <h3 style="color: {res['risk_color']}; margin: 0; font-weight: 700;">{res['risk_level']}</h3>
                <p style="color: #334155; margin: 6px 0 0 0; font-size: 0.95rem;">{res['explanation']}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<h3 class="dashboard-header">💡 Customer Profile Insights & SHAP Drivers</h3>', unsafe_allow_html=True)

        obs_col, shap_col = st.columns([1, 1])

        with obs_col:
            st.markdown("##### Key Profile Observations")
            tenure_msg = f"• **Account Tenure**: Customer is in onboarding stage ({tenure} months)." if tenure <= 3 else (f"• **Account Tenure**: Moderate account history ({tenure} months)." if tenure <= 12 else f"• **Account Tenure**: Established platform tenure ({tenure} months).")
            recency_msg = f"• **Recency**: High inactivity period ({days_since_last_order} days since last order)." if days_since_last_order > 30 else f"• **Recency**: Active customer engagement ({days_since_last_order} days since last order)."
            complain_msg = "• **Complaint History**: Active formal complaint logged on record." if complain == 1 else "• **Complaint History**: No active complaints logged."
            satisfaction_msg = f"• **Satisfaction**: Rated {satisfaction_score}/5 stars."

            st.markdown(f"""
            <div class="insight-box">
                {tenure_msg}<br><br>{recency_msg}<br><br>{complain_msg}<br><br>{satisfaction_msg}
            </div>
            """, unsafe_allow_html=True)

        top_shap_features = {}

        with shap_col:
            st.markdown("##### Local Feature Contribution (SHAP Explanation)")
            try:
                steps_dict = dict(model_pipeline.steps)
                
                fe_step = steps_dict.get('feature_engineer', steps_dict.get('fe'))
                prep_step = steps_dict.get('preprocessor', steps_dict.get('prep'))
                clf_step = steps_dict.get('classifier', steps_dict.get('model', model_pipeline[-1]))

                engineered_df = fe_step.transform(raw_input_df) if fe_step is not None else raw_input_df.copy()

                if prep_step is not None:
                    processed_input = prep_step.transform(engineered_df)
                    raw_feature_names = prep_step.get_feature_names_out()
                    clean_feature_names = [f.split('__')[-1] for f in raw_feature_names]
                else:
                    processed_input = engineered_df.values
                    clean_feature_names = engineered_df.columns.tolist()

                processed_df = pd.DataFrame(processed_input, columns=clean_feature_names)

                explainer = shap.TreeExplainer(clf_step)
                shap_vals = explainer(processed_df)
                shap_val_instance = shap_vals[0, :, 1] if len(shap_vals.shape) == 3 else shap_vals[0]

                top_indices = np.argsort(np.abs(shap_val_instance.values))[::-1][:5]
                for idx in top_indices:
                    top_shap_features[clean_feature_names[idx]] = round(float(shap_val_instance.values[idx]), 4)

                fig, ax = plt.subplots(figsize=(6, 4))
                shap.plots.waterfall(shap_val_instance, max_display=7, show=False)
                plt.title("Local Feature Contribution (SHAP)", fontsize=10, fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            except Exception as shap_err:
                st.warning(f"⚠️ Local SHAP waterfall explanation could not be rendered: {str(shap_err)}")

        st.markdown("---")
        st.markdown('<h3 class="dashboard-header">📄 Export Individual Prediction Report</h3>', unsafe_allow_html=True)

        report_col1, report_col2 = st.columns(2)

        json_report_data = {
            "report_metadata": {
                "model_name": metadata.get("model_name", "Random Forest Pipeline"),
                "model_version": metadata.get("model_version", "1.0.0"),
                "classification_threshold": threshold
            },
            "customer_profile_input": raw_input_df.to_dict(orient='records')[0],
            "prediction_summary": {
                "churn_probability": round(res['probability'], 4),
                "predicted_churn_class": int(res['predicted_class']),
                "risk_level": res['risk_level']
            },
            "top_shap_feature_drivers": top_shap_features
        }

        csv_report_df = raw_input_df.copy()
        csv_report_df['Churn_Probability'] = round(res['probability'], 4)
        csv_report_df['Predicted_Churn_Class'] = int(res['predicted_class'])
        csv_report_df['Risk_Level'] = res['risk_level']
        csv_report_df['Model_Threshold'] = threshold

        with report_col1:
            st.download_button(
                label="📥 Download JSON Report",
                data=json.dumps(json_report_data, indent=4),
                file_name="customer_churn_report.json",
                mime="application/json",
                use_container_width=True
            )

        with report_col2:
            st.download_button(
                label="📥 Download CSV Report",
                data=csv_report_df.to_csv(index=False).encode('utf-8'),
                file_name="customer_churn_report.csv",
                mime="text/csv",
                use_container_width=True
            )