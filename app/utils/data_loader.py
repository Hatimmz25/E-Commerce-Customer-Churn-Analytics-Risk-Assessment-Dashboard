import os
import pandas as pd
import streamlit as st

@st.cache_data
def load_raw_dataset(file_path='../../notebook/data/data_ecommerce_customer_churn.csv'):
    """Loads and standardizes the raw E-commerce Customer Churn dataset."""
    if not os.path.exists(file_path):
        file_path = 'data_ecommerce_customer_churn.csv'
        
    if not os.path.exists(file_path):
        return None
        
    try:
        df = pd.read_csv(file_path)
        if 'PreferedOrderCat' in df.columns:
            df['PreferedOrderCat'] = df['PreferedOrderCat'].replace({'Mobile': 'Mobile Phone'})
        return df
    except Exception:
        return None