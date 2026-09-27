import os
import sys

# Ensure root directory is at the head of sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
from app.utils.model_loader import load_assets
from app.components.single_prediction_view import render_single_prediction
from app.components.dataset_insights_view import render_dataset_insights
from app.components.model_performance_view import render_model_performance
from app.components.batch_prediction_view import render_batch_prediction

# Page Configuration
st.set_page_config(
    page_title="E-Commerce Churn Analytics & Risk Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
css_path = os.path.join(ROOT_DIR, 'app', 'styles', 'custom.css')
if os.path.exists(css_path):
    with open(css_path, 'r') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Asset Loading
model_pipeline, threshold, metadata, df_raw, df_comp, X_test, y_test, load_errors = load_assets()

if load_errors:
    for err in load_errors:
        st.warning(f"⚠️ {err}")
    if model_pipeline is None:
        st.error("❌ Critical Pipeline Error: Model pipeline failed to load.")
        st.stop()

safe_threshold = float(threshold) if threshold is not None else 0.45

# Header & Sidebar Navigation
st.markdown('<h1 class="dashboard-header">Customer Churn Analytics & Risk Assessment</h1>', unsafe_allow_html=True)
st.markdown('<p class="dashboard-subtitle">Enterprise Machine Learning Pipeline & Real-Time Churn Risk Engine</p>', unsafe_allow_html=True)

st.sidebar.markdown("### Model Information")
st.sidebar.markdown(f"**Model Type**: `{metadata.get('model_name', 'Random Forest Pipeline')}`")
st.sidebar.markdown(f"**Version**: `{metadata.get('model_version', '1.0.0')}`")
st.sidebar.markdown(f"**Optimal Threshold**: `{safe_threshold:.1%}`")

if df_raw is not None:
    st.sidebar.markdown(f"**Dataset Size**: `{len(df_raw):,} records`")

st.sidebar.markdown("---")
st.sidebar.markdown("### Risk Cutoffs")
st.sidebar.caption(f"""
- **LOW RISK**: Probability < {safe_threshold/2:.1%}
- **MEDIUM RISK**: {safe_threshold/2:.1%} ≤ Probability < {safe_threshold:.1%}
- **HIGH RISK**: Probability ≥ {safe_threshold:.1%}
""")

# Dashboard Tabs
main_tab1, main_tab2, main_tab3, main_tab4 = st.tabs([
    "🎯 Single Risk Prediction", 
    "📊 Dataset Insights & EDA", 
    "📈 Model Performance",
    "📁 Batch Prediction"
])

with main_tab1:
    render_single_prediction(model_pipeline, safe_threshold, metadata)

with main_tab2:
    render_dataset_insights(df_raw)

with main_tab3:
    render_model_performance(model_pipeline, safe_threshold, df_comp, X_test, y_test)

with main_tab4:
    render_batch_prediction(model_pipeline, safe_threshold)