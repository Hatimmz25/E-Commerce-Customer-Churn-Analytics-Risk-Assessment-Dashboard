import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def render_dataset_insights(df_raw):
    st.markdown('<h3 class="dashboard-header">📊 Dataset Exploratory Analytics</h3>', unsafe_allow_html=True)
    st.markdown("Interactive analysis of customer retention patterns derived from the full dataset.")

    if df_raw is None:
        st.warning("⚠️ **Dataset Unavailable**: `data/data_ecommerce_customer_churn.csv` was not found. Dataset insights tab cannot be rendered.")
        return

    # Key Statistics Metrics Row
    total_cust = len(df_raw)
    total_churned = int(df_raw['Churn'].sum()) if 'Churn' in df_raw.columns else 0
    overall_churn_rate = (total_churned / total_cust) * 100 if total_cust > 0 else 0

    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Total Observed Customers", f"{total_cust:,}")
    kpi2.metric("Total Churned Customers", f"{total_churned:,}")
    kpi3.metric("Overall Dataset Churn Rate", f"{overall_churn_rate:.2f}%")

    st.markdown("---")

    # Row 1 Charts: Churn Distribution & Churn by Order Category
    r1_c1, r1_c2 = st.columns(2)

    with r1_c1:
        st.markdown("##### 1. Churn Class Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        churn_counts = df_raw['Churn'].value_counts().reset_index()
        churn_counts.columns = ['Churn_Label', 'Count']
        churn_counts['Churn_Label'] = churn_counts['Churn_Label'].map({0: 'Retained (0)', 1: 'Churned (1)'})
        
        sns.barplot(data=churn_counts, x='Churn_Label', y='Count', hue='Churn_Label', palette=['#2b5c8f', '#d95f02'], ax=ax, legend=False)
        ax.set_ylabel("Customer Count")
        ax.set_xlabel("")
        for p in ax.patches:
            height = p.get_height()
            if not np.isnan(height):
                ax.annotate(f'{int(height):,}', (p.get_x() + p.get_width() / 2., height),
                            ha='center', va='bottom', xytext=(0, 3), textcoords='offset points')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with r1_c2:
        st.markdown("##### 2. Churn Rate by Preferred Order Category")
        if 'PreferedOrderCat' in df_raw.columns:
            cat_churn = df_raw.groupby('PreferedOrderCat')['Churn'].agg(['count', 'mean']).reset_index()
            cat_churn['Churn Rate (%)'] = cat_churn['mean'] * 100
            cat_churn = cat_churn.sort_values(by='Churn Rate (%)', ascending=False)
            
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(data=cat_churn, x='Churn Rate (%)', y='PreferedOrderCat', hue='PreferedOrderCat', palette='Reds_r', ax=ax, legend=False)
            ax.set_xlabel("Churn Rate (%)")
            ax.set_ylabel("")
            for p in ax.patches:
                width = p.get_width()
                if not np.isnan(width):
                    ax.annotate(f'{width:.1f}%', (width, p.get_y() + p.get_height() / 2.),
                                ha='left', va='center', xytext=(4, 0), textcoords='offset points')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    st.markdown("---")

    # Row 2 Charts: Marital Status & Complaint Status
    r2_c1, r2_c2 = st.columns(2)

    with r2_c1:
        st.markdown("##### 3. Churn Rate by Marital Status")
        if 'MaritalStatus' in df_raw.columns:
            marital_churn = df_raw.groupby('MaritalStatus')['Churn'].agg(['count', 'mean']).reset_index()
            marital_churn['Churn Rate (%)'] = marital_churn['mean'] * 100
            marital_churn = marital_churn.sort_values(by='Churn Rate (%)', ascending=False)

            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(data=marital_churn, x='MaritalStatus', y='Churn Rate (%)', hue='MaritalStatus', palette='Blues_r', ax=ax, legend=False)
            ax.set_ylabel("Churn Rate (%)")
            ax.set_xlabel("")
            for p in ax.patches:
                height = p.get_height()
                if not np.isnan(height):
                    ax.annotate(f'{height:.1f}%', (p.get_x() + p.get_width() / 2., height),
                                ha='center', va='bottom', xytext=(0, 3), textcoords='offset points')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with r2_c2:
        st.markdown("##### 4. Churn Rate by Complaint Status")
        if 'Complain' in df_raw.columns:
            df_temp = df_raw.copy()
            df_temp['Complaint_Label'] = df_temp['Complain'].map({0: 'No Complaint', 1: 'Logged Complaint'})
            comp_churn = df_temp.groupby('Complaint_Label')['Churn'].agg(['count', 'mean']).reset_index()
            comp_churn['Churn Rate (%)'] = comp_churn['mean'] * 100

            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(data=comp_churn, x='Complaint_Label', y='Churn Rate (%)', hue='Complaint_Label', palette=['#2b5c8f', '#dc2626'], ax=ax, legend=False)
            ax.set_ylabel("Churn Rate (%)")
            ax.set_xlabel("")
            for p in ax.patches:
                height = p.get_height()
                if not np.isnan(height):
                    ax.annotate(f'{height:.1f}%', (p.get_x() + p.get_width() / 2., height),
                                ha='center', va='bottom', xytext=(0, 3), textcoords='offset points')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()