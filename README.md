# 🛒 E-Commerce Customer Churn Analytics & Risk Assessment Platform

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://e-commerce-customer-churn-analytics-risk-assessment-dashboard.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4.1-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-green.svg)](https://xgboost.readthedocs.io/)

An end-to-end Machine Learning web application designed to predict, evaluate, and explain customer churn risk for e-commerce platforms. The platform features an interactive **Streamlit Dashboard** combining real-time individual risk assessment, SHAP-driven local explainability, batch inference capabilities, and population-level exploratory data analytics (EDA).

🔗 **Live Demo:** [E-Commerce Customer Churn Analytics Dashboard](https://e-commerce-customer-churn-analytics-risk-assessment-dashboard.streamlit.app/)

---

## 📋 Table of Contents
1. [Project Overview](#-project-overview)
2. [Business Problem](#-business-problem)
3. [Dataset & Features](#-dataset--features)
4. [Machine Learning Pipeline](#-machine-learning-pipeline)
5. [Models Evaluated & Performance](#-models-evaluated--performance)
6. [Threshold Optimization & Explainability (SHAP)](#-threshold-optimization--explainability-shap)
7. [Streamlit Application Architecture](#-streamlit-application-architecture)
8. [Project Structure](#-project-structure)
9. [Installation & Setup](#-installation--setup)
10. [Running the Application](#-running-the-application)
11. [Technologies Used](#-technologies-used)
12. [License](#-license)

---

## 🎯 1. Project Overview
Customer retention is a core economic driver for e-commerce businesses. Acquiring new customers is significantly more expensive than retaining existing ones. This project builds a production-ready machine learning pipeline and interactive web application to:
* Predict the exact probability that a given customer will churn.
* Classify customers into customized risk tiers (**Low**, **Medium**, and **High Risk**).
* Provide granular, transparent local explainability using **SHAP values** so retention teams know *why* a customer is at risk.
* Process single-profile forms and large batch CSV datasets seamlessly.

---

## 💼 2. Business Problem
E-commerce companies often suffer from silent churn—customers drifting away due to long delivery distances, unsatisfied product categories, or unresolved complaints. By deploying a predictive retention system, marketing and customer success teams can proactively allocate incentives or support to high-risk segments before churn occurs, directly protecting recurring platform revenue.

---

## 📊 3. Dataset & Features
The model is trained on a structured E-Commerce Customer Churn dataset containing demographic, behavioral, and transactional metrics.

### Target Variable
* **`Churn`**: Binary indicator (`1` = Churn / Customer left, `0` = Retained / Active).

### Input Features
| Feature Name | Data Type | Description |
| :--- | :--- | :--- |
| **`Tenure`** | Numerical | Number of months the customer has been with the platform |
| **`WarehouseToHome`** | Numerical | Distance in kilometers from the warehouse to the customer's address |
| **`NumberOfDeviceRegistered`** | Numerical | Total number of registered devices linked to the customer account |
| **`PreferedOrderCat`** | Categorical | Customer's preferred product order category (e.g., Laptop & Accessory, Mobile Phone, Fashion) |
| **`SatisfactionScore`** | Numerical | Customer feedback satisfaction rating on a scale from 1 to 5 |
| **`MaritalStatus`** | Categorical | Marital status of the customer (Single, Married, Divorced) |
| **`NumberOfAddress`** | Numerical | Number of distinct delivery addresses associated with the user account |
| **`Complain`** | Numerical | Binary indicator of whether the customer has lodged a complaint recently (`0` or `1`) |
| **`DaySinceLastOrder`** | Numerical | Number of days elapsed since the customer's last completed purchase |
| **`CashbackAmount`** | Numerical | Average cashback amount earned by the customer across transactions |

> **Note on Privacy:** No personally identifiable information (PII) such as Customer IDs, raw names, or email addresses are exposed or stored in the model inference pipeline.

---

## 🔄 4. Machine Learning Pipeline
The pipeline is constructed using custom scikit-learn transformers to prevent data leakage during cross-validation and ensure zero manual transformations during production inference:
1. **Data Understanding & EDA**: Statistical profiling and distribution checks (`01_data_understanding.ipynb`).
2. **Data Cleaning & Preprocessing**: Handling string anomalies and imputing missing values (`02_preprocessing.ipynb`).
3. **Custom Feature Engineering (`EcommerceFeatureEngineer`)**:
   * `CashbackPerTenure`: Ratio of cashback earned relative to account tenure.
   * `InactivityRatio`: Interaction modeling days since last order against tenure duration.
   * `HighRiskComplain`: Flagging accounts with active complaints.
   * `TenureStage`: Customer lifecycle segmentation (New, Established, Loyal).
4. **Model Training & Benchmarking**: Comparing baseline classifiers (`03_model_training.ipynb`).
5. **Hyperparameter Tuning & Optimization**: RandomizedSearchCV tuning (`04_model_tuning.ipynb`).
6. **Threshold Optimization**: Maximizing F1-score across probability decision boundaries.
7. **Model Explainability & Persistence**: SHAP values extraction and Joblib model serialization (`05_model_evaluation.ipynb`).

---

## 📈 5. Models Evaluated & Performance
Multiple candidate classifiers were evaluated during training, with **Random Forest** and **XGBoost** selected for optimal generalization on tabular metrics.

| Model Pipeline | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 84.1% | 0.652 | 0.710 | 0.680 | 0.892 |
| Decision Tree | 91.2% | 0.814 | 0.802 | 0.808 | 0.865 |
| **Random Forest (Production)** | **94.2%** | **0.875** | **0.854** | **0.864** | **0.961** |
| XGBoost Classifier | 93.8% | 0.862 | 0.849 | 0.855 | 0.958 |

---

## ⚖️ 6. Threshold Optimization & Explainability (SHAP)
* **Optimal Threshold Tuning**: Rather than using the default probability threshold of $t = 0.50$, optimization over validation folds yielded an optimal decision threshold of **$t = 0.45$**. This strategically captures higher recall for high-risk churning customers without causing excessive false positive mitigation costs.
* **SHAP Explainability**: Local interpretation integrates **SHAP (SHapley Additive exPlanations)** to generate waterfall and bar visualizations, breaking down exactly which features pushed a specific customer toward churn or retention.

---

## 🖥️ 7. Streamlit Application Architecture
The dashboard is structured into four interactive tabs:
1. **🎯 Single Risk Prediction**: Real-time form input evaluating individual risk tiers, probability scores, automated rule-based observations, and SHAP waterfall graphs.
2. **📊 Dataset Insights & EDA**: Population-level visualizations exploring churn distribution across cashback ranges, satisfaction scores, marital status, and order categories.
3. **📈 Model Performance**: Comparative metrics table, confusion matrix heatmaps, and ROC curves.
4. **📁 Batch Prediction**: Bulk CSV file upload engine generating predictions and downloadable batch CSV results.

---

## 📂 8. Project Structure
```text
ecommerce-customer-churn/
├── app/
│   ├── app.py                      # Main Streamlit application entry point
│   ├── components/                 # Modular dashboard UI views
│   │   ├── single_prediction_view.py
│   │   ├── dataset_insights_view.py
│   │   ├── model_performance_view.py
│   │   └── batch_prediction_view.py
│   ├── utils/                      # Helper modules & model loader
│   │   └── model_loader.py
│   └── styles/                     # Custom CSS UI styling
│       └── custom.css
├── data/                           # Dataset storage directory
│   ├── README.md
│   └── data_ecommerce_customer_churn.csv (User provided)
├── models/                         # Serialized production assets
│   ├── churn_model.joblib
│   ├── churn_threshold.joblib
│   ├── pipeline_metadata.json
│   └── tuned_model_comparison.csv
├── notebooks/                      # End-to-end Jupyter notebooks
│   ├── 01_data_understanding.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_model_tuning.ipynb
│   └── 05_model_evaluation.ipynb
├── requirements.txt                # Pinned dependencies
├── .gitignore
├── LICENSE
└── README.md

🚀 16. Installation
Clone the repository:

Bash
git clone [https://github.com/Hatimmz25/E-Commerce-Customer-Churn-Analytics-Risk-Assessment-Dashboard.git](https://github.com/Hatimmz25/E-Commerce-Customer-Churn-Analytics-Risk-Assessment-Dashboard.git)
cd E-Commerce-Customer-Churn-Analytics-Risk-Assessment-Dashboard
Create and activate a virtual environment:

Bash
python -m venv venv
source venv/bin/activate      # On macOS/Linux
venv\Scripts\activate         # On Windows
Install dependencies:

Bash
pip install -r requirements.txt
💻 17. Running the Application
To launch the Streamlit application locally, execute:

Bash
streamlit run app/app.py
Upon execution, open your web browser at http://localhost:8501.

🛠️ 18. Technologies Used
Programming Language: Python 3.10+

Web Application Framework: Streamlit

Machine Learning Frameworks: scikit-learn, XGBoost

Model Explainability: SHAP

Data Manipulation & Analysis: Pandas, NumPy

Data Visualization: Matplotlib, Seaborn

Model Serialization: Joblib
