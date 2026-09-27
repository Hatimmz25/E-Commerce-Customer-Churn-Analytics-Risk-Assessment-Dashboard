import os
import sys
import types
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin

# 1. Custom Feature Engineer Transformer (Must match exact class structure used during model fitting)
class EcommerceFeatureEngineer(BaseEstimator, TransformerMixin):
    """Encapsulates domain feature engineering for the E-Commerce churn pipeline."""
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        
        # Feature 1: CashbackPerTenure
        if 'CashbackAmount' in X_out.columns and 'Tenure' in X_out.columns:
            tenure_safe = X_out['Tenure'].fillna(0)
            X_out['CashbackPerTenure'] = X_out['CashbackAmount'] / (tenure_safe + 1.0)
            
        # Feature 2: InactivityRatio
        if 'DaySinceLastOrder' in X_out.columns and 'Tenure' in X_out.columns:
            tenure_safe = X_out['Tenure'].fillna(0)
            days_safe = X_out['DaySinceLastOrder'].fillna(0)
            X_out['InactivityRatio'] = days_safe / (tenure_safe * 30.0 + 1.0)
            
        # Feature 3: HighRiskComplain
        if 'Complain' in X_out.columns and 'SatisfactionScore' in X_out.columns:
            complain_flag = X_out['Complain'].fillna(0)
            satisfaction_val = X_out['SatisfactionScore'].fillna(3)
            X_out['HighRiskComplain'] = ((complain_flag == 1) & (satisfaction_val <= 2)).astype(int)
            
        # Feature 4: TenureStage
        if 'Tenure' in X_out.columns:
            def bin_tenure(val):
                if pd.isna(val) or val <= 3:
                    return 'Onboarding'
                elif val <= 12:
                    return 'Early'
                elif val <= 24:
                    return 'Established'
                else:
                    return 'Loyal'
            X_out['TenureStage'] = X_out['Tenure'].apply(bin_tenure)
            
        return X_out

# 2. Namespace Alias Injection: Fix joblib deserialization when model was saved from __main__
if 'main' not in sys.modules:
    sys.modules['main'] = types.ModuleType('main')

sys.modules['main'].EcommerceFeatureEngineer = EcommerceFeatureEngineer

if '__main__' in sys.modules:
    sys.modules['__main__'].EcommerceFeatureEngineer = EcommerceFeatureEngineer

# Strict Required Schema for Raw E-Commerce Customer Inputs
REQUIRED_FEATURES = [
    'Tenure', 'WarehouseToHome', 'NumberOfDeviceRegistered',
    'PreferedOrderCat', 'SatisfactionScore', 'MaritalStatus',
    'NumberOfAddress', 'Complain', 'DaySinceLastOrder', 'CashbackAmount'
]

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@st.cache_resource
def load_assets():
    """Loads and caches production model pipelines, thresholds, metadata, and evaluation data."""
    errors = []
    
    # 1. Model Pipeline Loading
    model_path = os.path.join(ROOT_DIR, 'models', 'churn_model.joblib')
    if not os.path.exists(model_path):
        errors.append(f"Model file (`{model_path}`) is missing.")
        model_pipeline = None
    else:
        try:
            model_pipeline = joblib.load(model_path)
        except Exception as e:
            errors.append(f"Model file (`{model_path}`) could not be loaded: {str(e)}")
            model_pipeline = None

    # 2. Optimal Threshold Loading
    threshold_path = os.path.join(ROOT_DIR, 'models', 'churn_threshold.joblib')
    threshold = 0.45
    if os.path.exists(threshold_path):
        try:
            loaded_thresh = joblib.load(threshold_path)
            threshold = float(loaded_thresh)
        except Exception:
            threshold = 0.45

    # 3. Pipeline Metadata
    metadata_path = os.path.join(ROOT_DIR, 'models', 'pipeline_metadata.json')
    metadata = {"model_name": "Random Forest Pipeline", "model_version": "1.0.0", "training_date": "N/A"}
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        except Exception:
            pass

    # 4. Raw Dataset Loading
    dataset_path = os.path.join(ROOT_DIR, 'data', 'data_ecommerce_customer_churn.csv')
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(ROOT_DIR, 'data_ecommerce_customer_churn.csv')
        
    df_raw = None
    if os.path.exists(dataset_path):
        try:
            df_raw = pd.read_csv(dataset_path)
            if 'PreferedOrderCat' in df_raw.columns:
                df_raw['PreferedOrderCat'] = df_raw['PreferedOrderCat'].replace({'Mobile': 'Mobile Phone'})
        except Exception:
            df_raw = None

    # 5. Model Comparison Data
    comp_path = os.path.join(ROOT_DIR, 'models', 'tuned_model_comparison.csv')
    if not os.path.exists(comp_path):
        comp_path = os.path.join(ROOT_DIR, 'models', 'model_comparison.csv')
        
    df_comp = None
    if os.path.exists(comp_path):
        try:
            df_comp = pd.read_csv(comp_path)
        except Exception:
            df_comp = None

    # 6. Test Arrays / Dynamic Fallback
    test_path = os.path.join(ROOT_DIR, 'data', 'ecom_processed_data.npz')
    X_test, y_test = None, None
    if os.path.exists(test_path):
        try:
            proc_data = np.load(test_path, allow_pickle=True)
            X_test, y_test = proc_data['X_test'], proc_data['y_test']
        except Exception:
            pass

    if (X_test is None or y_test is None) and df_raw is not None and 'Churn' in df_raw.columns:
        try:
            X_test = df_raw[REQUIRED_FEATURES].copy()
            y_test = df_raw['Churn'].values
        except Exception:
            pass

    return model_pipeline, threshold, metadata, df_raw, df_comp, X_test, y_test, errors