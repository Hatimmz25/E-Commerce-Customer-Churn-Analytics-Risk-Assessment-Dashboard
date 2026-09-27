import pandas as pd
import numpy as np

def get_risk_tier(p, threshold):
    """Calculates standardized risk level based on continuous probability and dynamic cutoff threshold."""
    if p >= threshold:
        return "HIGH RISK"
    elif p >= (threshold / 2.0):
        return "MEDIUM RISK"
    else:
        return "LOW RISK"

def predict_single(model_pipeline, raw_input_df, threshold):
    """Performs single customer prediction using the production scikit-learn pipeline."""
    proba = float(model_pipeline.predict_proba(raw_input_df)[:, 1][0])
    pred_class = 1 if proba >= threshold else 0
    risk_level = get_risk_tier(proba, threshold)
    
    if risk_level == "HIGH RISK":
        risk_style = "risk-card-high"
        risk_color = "#dc2626"
        explanation = f"Estimated churn probability ({proba:.1%}) meets or exceeds the decision threshold ({threshold:.1%}). Immediate proactive retention action recommended."
    elif risk_level == "MEDIUM RISK":
        risk_style = "risk-card-medium"
        risk_color = "#d97706"
        explanation = f"Estimated churn probability ({proba:.1%}) is approaching threshold ({threshold:.1%}). Elevated churn risk detected; monitor engagement closely."
    else:
        risk_style = "risk-card-low"
        risk_color = "#16a34a"
        explanation = f"Estimated churn probability ({proba:.1%}) is well below the decision threshold ({threshold:.1%}). Low retention risk."

    return {
        "probability": proba,
        "predicted_class": pred_class,
        "risk_level": risk_level,
        "risk_style": risk_style,
        "risk_color": risk_color,
        "explanation": explanation
    }