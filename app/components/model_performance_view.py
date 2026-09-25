import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

def render_model_performance(model_pipeline, threshold, df_comp, X_test, y_test):
    st.markdown('<h3 class="dashboard-header">📈 Model Training & Evaluation Performance</h3>', unsafe_allow_html=True)
    st.markdown("Comparative evaluation metrics for candidate machine learning models trained on the E-Commerce Customer Churn dataset.")

    if df_comp is None or X_test is None or y_test is None:
        st.warning("⚠️ **Evaluation Artifacts Missing**: Model comparison logs or test datasets were not found.")
        return

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("Production Model", "Random Forest")
    m_col2.metric("Optimal Threshold", f"{threshold:.1%}")
    m_col3.metric("Held-Out Test Set", f"{len(y_test):,} samples")
    m_col4.metric("Evaluation Metric", "F1 Score (Balanced)")

    st.markdown("---")

    st.markdown("##### 1. Machine Learning Model Comparison Table")
    df_display = df_comp.copy()
    for col in ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']:
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(lambda x: f"{x:.4f}" if isinstance(x, (float, int)) else x)

    st.dataframe(df_display, use_container_width=True)

    st.markdown("---")

    st.markdown("##### 2. Selected Production Model Performance (Random Forest Pipeline)")
    
    try:
        y_prob_test = model_pipeline.predict_proba(X_test)[:, 1]
        y_pred_test = (y_prob_test >= threshold).astype(int)

        cm = confusion_matrix(y_test, y_pred_test)
        tn, fp, fn, tp = cm.ravel()

        pm1, pm2, pm3, pm4 = st.columns(4)
        pm1.metric("Test Accuracy", f"{(tn+tp)/len(y_test):.2%}")
        pm2.metric("Test Precision", f"{tp/(tp+fp):.2%}" if (tp+fp) > 0 else "N/A")
        pm3.metric("Test Recall (Sensitivity)", f"{tp/(tp+fn):.2%}" if (tp+fn) > 0 else "N/A")
        pm4.metric("Test F1 Score", f"{2*tp/(2*tp+fp+fn):.4f}" if (2*tp+fp+fn) > 0 else "N/A")

        st.markdown("<br>", unsafe_allow_html=True)

        fig_col1, fig_col2 = st.columns(2)

        with fig_col1:
            st.markdown(f"##### Confusion Matrix (Decision Threshold = {threshold:.2f})")
            fig, ax = plt.subplots(figsize=(6, 4.5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                        xticklabels=['Retained (0)', 'Churned (1)'],
                        yticklabels=['Retained (0)', 'Churned (1)'], ax=ax)
            ax.set_xlabel('Predicted Class')
            ax.set_ylabel('True Class')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with fig_col2:
            st.markdown("##### Receiver Operating Characteristic (ROC) Curve")
            fpr, tpr, _ = roc_curve(y_test, y_prob_test)
            roc_auc_val = auc(fpr, tpr)

            fig, ax = plt.subplots(figsize=(6, 4.5))
            ax.plot(fpr, tpr, color='#2b5c8f', lw=2.5, label=f'Random Forest (AUC = {roc_auc_val:.4f})')
            ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
            ax.set_xlabel('False Positive Rate (1 - Specificity)')
            ax.set_ylabel('True Positive Rate (Sensitivity)')
            ax.legend(loc='lower right')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    except Exception:
        st.error("⚠️ Evaluation graphics could not be rendered from test set objects.")