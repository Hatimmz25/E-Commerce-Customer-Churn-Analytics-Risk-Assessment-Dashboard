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
        # Standardize categorical input value
        cat_value = 'Mobile Phone' if preferred_cat == 'Mobile' else preferred_cat

        raw_input_df = pd.DataFrame([{
            'Tenure': tenure, 'WarehouseToHome': warehouse_dist,
            'NumberOfDeviceRegistered': num_devices, 'PreferedOrderCat': cat_value,
            'SatisfactionScore': satisfaction_score, 'MaritalStatus': marital_status,
            'NumberOfAddress': num_addresses, 'Complain': complain,
            'DaySinceLastOrder': days_since_last_order, 'CashbackAmount': cashback
        }])

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

        st.info("ℹ️ **Interpretation Note**: Continuous probabilities reflect historical model pattern scoring and relative risk likelihood.")

        st.markdown("---")
        st.markdown('<h3 class="dashboard-header">💡 Customer Profile Insights & Explanation</h3>', unsafe_allow_html=True)

        obs_col, shap_col = st.columns([1, 1])

        with obs_col:
            st.markdown("##### Key Profile Observations")
            tenure_msg = f"• **Account Tenure**: Customer is in early onboarding stage ({tenure} months)." if tenure <= 3 else (f"• **Account Tenure**: Customer has moderate account history ({tenure} months)." if tenure <= 12 else f"• **Account Tenure**: Customer exhibits established platform tenure ({tenure} months).")
            recency_msg = f"• **Recency**: Customer has a relatively long inactivity period since last order ({days_since_last_order} days)." if days_since_last_order > 30 else (f"• **Recency**: Moderate inactivity period recorded ({days_since_last_order} days since order)." if days_since_last_order > 10 else f"• **Recency**: Active purchasing behavior with recent order ({days_since_last_order} days ago).")
            complain_msg = "• **Complaint History**: Active formal complaint logged on record." if complain == 1 else "• **Complaint History**: No recent formal complaints logged."
            satisfaction_msg = f"• **Satisfaction Level**: Self-reported rating of {satisfaction_score}/5."
            cashback_msg = f"• **Financial Rewards**: Accumulated ${cashback:.2f} in average cashback rewards."
            device_msg = f"• **Platform Access**: Accesses platform across {num_devices} registered device(s)."

            st.markdown(f"""
            <div class="insight-box">
                {tenure_msg}<br><br>{recency_msg}<br><br>{complain_msg}<br><br>{satisfaction_msg}<br><br>{cashback_msg}<br><br>{device_msg}
            </div>
            """, unsafe_allow_html=True)

        top_shap_features = {}

        with shap_col:
            st.markdown("##### Individual Model Prediction Drivers (SHAP Explanation)")
            try:
                # Safely inspect pipeline steps
                steps_dict = dict(model_pipeline.steps)
                
                # Identify steps
                fe_step = steps_dict.get('feature_engineer', steps_dict.get('fe'))
                prep_step = steps_dict.get('preprocessor', steps_dict.get('prep'))
                clf_step = steps_dict.get('classifier', steps_dict.get('model', model_pipeline[-1]))

                if fe_step is not None:
                    engineered_df = fe_step.transform(raw_input_df)
                else:
                    engineered_df = raw_input_df.copy()

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

                fig, ax = plt.subplots(figsize=(6, 4.5))
                shap.plots.waterfall(shap_val_instance, max_display=7, show=False)
                plt.title("Local Feature Contribution (SHAP Waterfall)", fontsize=10, fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                st.caption("🔍 **Explanation Note**: Visualizes model statistical feature contributions. Does not imply causality.")

            except Exception as shap_err:
                st.warning(f"⚠️ Local SHAP waterfall explanation could not be computed: {str(shap_err)}")

        st.markdown("---")
        st.markdown('<h3 class="dashboard-header">📄 Export Individual Prediction Report</h3>', unsafe_allow_html=True)

        report_col1, report_col2 = st.columns(2)

        json_report_data = {
            "report_metadata": {
                "model_name": metadata.get("model_name", "Random Forest Pipeline"),
                "model_version": metadata.get("model_version", "1.0.0"),
                "classification_threshold": threshold
            },
            "customer_profile_input": {
                "Tenure_Months": tenure, "MaritalStatus": marital_status,
                "SavedAddresses": num_addresses, "DaysSinceLastPurchase": days_since_last_order,
                "RegisteredDevices": num_devices, "SatisfactionScore": satisfaction_score,
                "LoggedComplaint": "Yes" if complain == 1 else "No", "PrimaryOrderCategory": preferred_cat,
                "AverageCashback_USD": cashback, "FulfillmentDistance_KM": warehouse_dist
            },
            "prediction_summary": {
                "churn_probability": round(res['probability'], 4),
                "predicted_churn_class": int(res['predicted_class']),
                "risk_level": res['risk_level'],
                "decision_threshold_applied": threshold
            },
            "top_model_feature_drivers_shap": top_shap_features
        }

        csv_report_df = pd.DataFrame([{
            'Tenure_Months': tenure, 'MaritalStatus': marital_status,
            'SavedAddresses': num_addresses, 'DaysSinceLastPurchase': days_since_last_order,
            'RegisteredDevices': num_devices, 'SatisfactionScore': satisfaction_score,
            'LoggedComplaint': "Yes" if complain == 1 else "No", 'PrimaryOrderCategory': preferred_cat,
            'AverageCashback_USD': cashback, 'FulfillmentDistance_KM': warehouse_dist,
            'Churn_Probability': round(res['probability'], 4), 'Predicted_Churn_Class': int(res['predicted_class']),
            'Risk_Level': res['risk_level'], 'Model_Threshold': threshold
        }])

        with report_col1:
            st.download_button(
                label="📥 Download Report as JSON File",
                data=json.dumps(json_report_data, indent=4),
                file_name="customer_churn_report.json",
                mime="application/json",
                use_container_width=True
            )

        with report_col2:
            st.download_button(
                label="📥 Download Report as CSV File",
                data=csv_report_df.to_csv(index=False).encode('utf-8'),
                file_name="customer_churn_report.csv",
                mime="text/csv",
                use_container_width=True
            )