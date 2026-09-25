# 🛒 E-Commerce Customer Churn Analytics & Risk Assessment Dashboard

An end-to-end, enterprise-grade Machine Learning solution and interactive Streamlit application designed to predict, analyze, and explain customer churn for e-commerce platforms.

---

## 📌 1. Project Overview

Customer retention is a vital driver of profitability and sustainable growth in the e-commerce sector. Acquiring a new customer can cost up to 5 to 25 times more than retaining an existing one. This project implements a production-ready Machine Learning pipeline that identifies customers at risk of churning, evaluates the business impact of retention interventions, and provides actionable, model-explainable insights using SHAP (SHapley Additive exPlanations).

The solution features an interactive, modern Streamlit analytics dashboard capable of real-time single-customer risk scoring, batch CSV predictions, exploratory dataset analytics, and comprehensive model performance tracking.

---

## 🎯 2. Business Problem

In e-commerce, customer churn is often non-contractual—customers simply cease placing orders without providing explicit cancellation notices. Identifying subtle indicators of disengagement (e.g., increased days since last order, low cashback accumulation, unaddressed complaints) allows retention teams to execute targeted promotional strategies prior to churn.

### Core Objectives:
1. **Early Identification**: Predict the continuous probability of customer churn before account abandonment.
2. **Optimal Decision Thresholding**: Balance the financial cost of false alarms (unnecessary promotional discounts) against undetected churn (loss of customer lifetime value).
3. **Model Explainability**: Provide transparent explanations for individual predictions to build operational trust and inform retention strategies.
4. **Operational Deployment**: Deliver a modular, fault-tolerant analytics application for business stakeholders.

---

## 📊 3. Dataset Overview

The project utilizes the **E-Commerce Customer Churn Dataset** (`data/data_ecommerce_customer_churn.csv`), comprising **5,630 customer records** and **20 raw attributes** covering account demographics, ordering habits, logistics, and behavioral feedback.

* **Total Records**: 5,630
* **Feature Count**: 19 Features + 1 Target Variable
* **Target Class Imbalance**: ~83.16% Retained (Class 0) vs. ~16.84% Churned (Class 1)

---

## 📋 4. Dataset Schema & Features

The dataset includes the following input features:

| Feature Name | Data Type | Description |
| :--- | :--- | :--- |
| `Tenure` | Numeric (`float64`) | Duration of customer relationship on the platform (in months). |
| `WarehouseToHome` | Numeric (`float64`) | Distance from fulfillment warehouse to customer delivery address (km). |
| `NumberOfDeviceRegistered` | Numeric (`int64`) | Total number of devices registered to the customer account. |
| `PreferedOrderCat` | Categorical (`object`) | Most frequently ordered product category (e.g., *Laptop & Accessory*, *Mobile Phone*, *Fashion*, *Grocery*, *Others*). |
| `SatisfactionScore` | Numeric (`int64`) | Self-reported customer satisfaction score ($1 = \text{Dissatisfied}$, $5 = \text{Highly Satisfied}$). |
| `MaritalStatus` | Categorical (`object`) | Marital status of customer (*Single*, *Married*, *Divorced*). |
| `NumberOfAddress` | Numeric (`int64`) | Total saved shipping addresses linked to the account. |
| `Complain` | Binary (`int64`) | Flag indicating whether a formal complaint was logged ($1 = \text{Yes}$, $0 = \text{No}$). |
| `DaySinceLastOrder` | Numeric (`float64`) | Days elapsed since the customer placed their most recent purchase. |
| `CashbackAmount` | Numeric (`float64`) | Total or average cashback rewards earned ($). |

---

## 🎯 5. Target Variable

The predictive target variable is **`Churn`**:
* **`0` (Retained)**: The customer remains actively engaged with the platform.
* **`1` (Churned)**: The customer has discontinued platform activity.

---

## 🛠️ 6. Data Preprocessing

To guarantee zero data leakage between training and evaluation phases, preprocessing transformations are constructed using a scikit-learn `ColumnTransformer` fitted exclusively on the 80% training split:

1. **Category Standardization**: Unifies inconsistent categorical labels (e.g., mapping string typos like `'Mobile'` to `'Mobile Phone'`).
2. **Missing Value Imputation**:
   * Numerical features (`Tenure`, `WarehouseToHome`, `DaySinceLastOrder`, etc.) are imputed using median values to handle right-skewness.
   * Categorical features are imputed using mode (most frequent value).
3. **Categorical Encoding**: `OneHotEncoder(handle_unknown='ignore')` transforms nominal variables (`PreferedOrderCat`, `MaritalStatus`) into binary vectors while ensuring robustness against unseen categories during deployment.
4. **Feature Scaling**: `StandardScaler()` standardizes numerical inputs for distance-sensitive algorithms like Logistic Regression.

---

## ⚙️ 7. Feature Engineering

Four domain-specific features were engineered to capture non-linear behavioral interactions and lifecycle stages:

1. **`CashbackPerTenure`** ($\frac{\text{CashbackAmount}}{\text{Tenure} + 1}$): Measures average cashback accumulation per month of tenure. Differentiates loyal reward earners from short-term promotional seekers.
2. **`InactivityRatio`** ($\frac{\text{DaySinceLastOrder}}{\text{Tenure} \times 30 + 1}$): Measures days inactive relative to the customer's total lifetime platform days.
3. **`HighRiskComplain`** ($\text{Complain} == 1 \text{ and } \text{SatisfactionScore} \le 2$): A binary flag highlighting severe dissatisfaction interactions.
4. **`TenureStage`**: Categorical lifecycle stage binning (`Onboarding` $\le 3\text{m}$, `Early` $3\text{--}12\text{m}$, `Established` $12\text{--}24\text{m}$, `Loyal` $> 24\text{m}$).

All feature engineering logic is encapsulated inside a custom, reproducible scikit-learn transformer (`EcommerceFeatureEngineer`).

---

## 🧪 8. Models Evaluated

Four classification algorithms were evaluated under an identical 80/20 stratified train/test split:

1. **Logistic Regression** (Linear baseline with balanced class weights)
2. **Decision Tree Classifier** (Non-linear tree baseline)
3. **Random Forest Classifier** (Ensemble tree bagging)
4. **XGBoost Classifier** (Gradient boosted decision trees)

---

## 🔍 9. Hyperparameter Tuning

Hyperparameter optimization was performed on the training set using **5-Fold Stratified Cross-Validation** with `RandomizedSearchCV` and `GridSearchCV`, targeting **F1-Score** optimization to address class imbalance.

* **Tuned Random Forest Parameters**:
  * `n_estimators`: 300
  * `max_depth`: 20
  * `min_samples_split`: 2
  * `min_samples_leaf`: 1
  * `max_features`: `'sqrt'`
  * `class_weight`: `'balanced'`

* **Tuned XGBoost Parameters**:
  * `n_estimators`: 300, `max_depth`: 6, `learning_rate`: 0.05, `subsample`: 0.8, `colsample_bytree`: 0.8, `gamma`: 0.1

---

## 🎚️ 10. Threshold Optimization

Rather than using a default `0.50` probability cutoff, decision threshold optimization was executed across the continuous range $0.10 \le t \le 0.90$. 

The optimal decision threshold was identified as **`0.45`** (or **45.0%**), which maximizes the test set **F1-Score** ($0.9534$), achieving an optimal trade-off between **Precision** ($98.86\%$) and **Recall** ($92.06\%$).

---

## 📊 11. Model Evaluation

Performance metrics evaluated on the held-out test set ($N = 1,126$):

| Model Candidate | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Tuned Random Forest (Selected)** | **98.49%** | **98.86%** | **92.06%** | **0.9534** | **0.9984** |
| Tuned XGBoost | 97.60% | 96.09% | 89.42% | 0.9264 | 0.9942 |
| Decision Tree | 97.42% | 91.53% | 93.12% | 0.9232 | 0.9570 |
| Tuned Logistic Regression | 81.35% | 46.13% | 80.95% | 0.5873 | 0.8879 |

---

## 🧠 12. SHAP Explainability

Model explainability is integrated using **SHAP (SHapley Additive exPlanations)** TreeExplainer:

* **Global Drivers**: `Tenure`, `CashbackAmount`, `Complain`, `DaySinceLastOrder`, and engineered ratios (`InactivityRatio`, `CashbackPerTenure`) represent the strongest overall predictors of churn across the customer population.
* **Local Waterfall Plots**: The Streamlit dashboard renders real-time SHAP waterfall charts for single-customer predictions, visualizing the exact features pushing probability toward or away from churn.
* **Causal Distinction**: All SHAP values are explicitly annotated as model feature contributions rather than direct causal claims.

---

## 🖥️ 13. Streamlit Analytics Application

The Streamlit dashboard (`app/app.py`) provides an executive interface structured across 4 core tabs:

1. **🎯 Single Risk Prediction**: Interactive customer form, visual probability meter, 3-tier risk banner (`LOW RISK`, `MEDIUM RISK`, `HIGH RISK`), rule-based profile observations, SHAP waterfall explanation, and downloadable PDF/JSON/CSV prediction reports.
2. **📊 Dataset Insights & EDA**: Interactive population analytics covering churn rates across order categories, marital status, satisfaction levels, complaints, and tenure stages.
3. **📈 Model Performance**: Comprehensive model comparison tables, test set confusion matrices, and ROC curves.
4. **📁 Batch Prediction Engine**: Automated CSV upload, schema validation, batch inference, risk tier assignment, and CSV export.

---

## 📁 14. Batch Prediction Feature

The Batch Prediction module allows stakeholders to evaluate entire customer cohorts simultaneously:
1. Validates uploaded CSV schema against required features.
2. Excludes target columns (`Churn`) or primary keys (`CustomerID`) automatically.
3. Passes raw inputs directly through the end-to-end production pipeline.
4. Generates a downloadable CSV complete with `Churn_Probability`, `Predicted_Churn_Class`, and assigned `Risk_Level`.

---

## 📂 15. Project Structure

```text
project/
│
├── app/
│   ├── app.py                      # Streamlit Application Entry Point
│   ├── components/                 # Modular Dashboard Views
│   │   ├── __init__.py
│   │   ├── single_prediction_view.py
│   │   ├── dataset_insights_view.py
│   │   ├── model_performance_view.py
│   │   └── batch_prediction_view.py
│   ├── utils/                      # Model Loading & Predictor Helpers
│   │   ├── __init__.py
│   │   ├── model_loader.py
│   │   ├── predictor.py
│   │   └── data_loader.py
│   └── styles/
│       └── custom.css              # Custom Styling Sheet
│
├── data/
│   └── data_ecommerce_customer_churn.csv  # Raw E-Commerce Dataset
│
├── models/
│   ├── churn_model.joblib          # Trained Production Model Pipeline Asset
│   ├── churn_threshold.joblib      # Saved Optimal Decision Threshold
│   └── pipeline_metadata.json      # Pipeline Configuration Metadata
│
├── notebook/
│   ├── 01_data_understanding.ipynb # EDA & Statistical Summaries
│   ├── 02_preprocessing.ipynb      # Feature Engineering & Pipeline Creation
│   ├── 03_model_training.ipynb     # Model Benchmarking & Selection
│   ├── 04_model_tuning.ipynb       # Hyperparameter & Threshold Optimization
│   └── 05_model_evaluation.ipynb   # SHAP Explainability & ROC Curves
│
├── requirements.txt                # Dependency Requirements File
└── README.md                       # Project Documentation


🚀 16. Installation
Clone the repository:

Bash
git clone [https://github.com/your-username/ecommerce-customer-churn.git](https://github.com/your-username/ecommerce-customer-churn.git)
cd ecommerce-customer-churn
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